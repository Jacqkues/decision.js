"""References from d3-lite's own runtime (vllm-sr/d3-lite, transformers==5.17.0):

    python scripts/d3-reference.py
      → test/fixtures/d3-prompts.json: token ids and option count of every question of test/fixtures/prompts.json
        (each state with each question), or the runtime's error; plus, for the first one, the fp32 hidden states
        after every layer at a few tokens (debugging)
      → bench/decision-index/reference-d3.json: the official (BF16) probabilities on the Decision Index sample
Probabilities are in option order ([yes, no] for a noul, as decision.js returns them)."""
import json, sys, time
from pathlib import Path
import torch
from transformers import AutoModel

root = Path(__file__).resolve().parent.parent
device = sys.argv[1] if len(sys.argv) > 1 else ("mps" if torch.backends.mps.is_available() else "cpu")
m = AutoModel.from_pretrained("vllm-sr/d3-lite", trust_remote_code=True, device=device)
rt = m.runtime
captured = []
orig = type(rt).probabilities
def spy(self, seqs, counts):
    captured.append((list(map(list, seqs)), list(counts)))
    return orig(self, seqs, counts)
type(rt).probabilities = spy


def probs(q, ans):
    if q["type"] == "noul":
        return [ans["noul"], 1 - ans["noul"]]
    if q["type"] == "score":
        return [ans["probabilities"][str(i)] for i in range(len(q["criteria"]))]
    return [ans["probabilities"][k] for k in q["criteria"]]


# ---- prompts
fx = json.load(open(root / "test/fixtures/prompts.json"))
cases = []
for si, s in enumerate(fx["states"]):
    for qi, q in enumerate(fx["questions"]):
        captured.clear()
        try:
            ans = m.system_one(state=s, questions={"q": q})["answers"]["q"]
        except Exception as e:  # noqa: BLE001
            ans = {"error": f"{type(e).__name__}: {e}"}
        if "error" in ans:
            cases.append({"state": si, "question": qi, "error": str(ans.get("message") or ans["error"])[:300]})
        else:
            seqs, counts = captured[-1]
            cases.append({"state": si, "question": qi, "ids": seqs[0], "options": counts[0], "probabilities": probs(q, ans)})
print(f"prompts: {sum('ids' in c for c in cases)} encoded, {sum('error' in c for c in cases)} refused", flush=True)

# ---- fp32 hidden states of the first encoded prompt, after the embedding and every layer
first = next(c for c in cases if "ids" in c)
lm = rt.backbone.language_model
lm32 = lm.float()
ids = torch.tensor([first["ids"]], device=lm32.embed_tokens.weight.device)
with torch.no_grad():
    out = lm32(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
picks = [0, 1, len(first["ids"]) // 2, len(first["ids"]) - 1]
debug = {"tokens": picks, "hidden": [[h[0, t].float().cpu().tolist() for t in picks] for h in out.hidden_states],
         "final": [out.last_hidden_state[0, t].float().cpu().tolist() for t in picks]}
lm.to(torch.bfloat16)
json.dump({"repo": "vllm-sr/d3-lite", "cases": cases, "debug": {"state": first["state"], "question": first["question"], **debug}},
          open(root / "test/fixtures/d3-prompts.json", "w"))

# ---- Decision Index sample
di = root / "bench/decision-index/sample.json"
if di.exists():
    sample = json.load(open(di))
    res, t0, ok, n = [], time.time(), 0, 0
    for i, r in enumerate(sample):
        ans = m.system_one(state=r["state"], questions=r["questions"])["answers"]
        res.append([probs(q, ans[name]) for name, q in r["questions"].items()])
        for (name, q), p in zip(r["questions"].items(), res[-1]):
            keys = list(q["criteria"]); ok += keys[max(range(len(p)), key=p.__getitem__)] == r["expected"][name]; n += 1
        if i % 40 == 0:
            print(f"{i}/{len(sample)} · {time.time() - t0:.0f} s", flush=True)
    print(f"vllm-sr/d3-lite: accuracy {ok}/{n} = {ok / n:.3f}")
    json.dump({"repo": "vllm-sr/d3-lite", "probabilities": res}, open(root / "bench/decision-index/reference-d3.json", "w"))
