"""References from Laya's own runtime (pip install laya), fp32:

    python scripts/laya-reference.py [repo] [subfolder]
      → test/fixtures/laya-prompts.json: the token ids and option markers of every question of
        test/fixtures/prompts.json (each state with each question; or the runtime's error)
      → bench/decision-index/reference-laya.json: probabilities on the Decision Index sample, if present
Probabilities are in option order ([yes, no] for a noul, as d1.js returns them)."""
import json, sys, time
from pathlib import Path
import torch, laya
from laya import agent as agent_mod

repo = sys.argv[1] if len(sys.argv) > 1 else "convaiinnovations/laya"
sub = sys.argv[2] if len(sys.argv) > 2 else None
root = Path(__file__).resolve().parent.parent
a = laya.load(repo, subfolder=sub, device="mps" if torch.backends.mps.is_available() else "cpu")
a.amp_enabled = False  # fp32 reference

captured = []
orig = agent_mod.Agent._encode_state
def spy(self, state, ids, internal, **kw):
    items = orig(self, state, ids, internal, **kw)
    captured.append(items)
    return items
agent_mod.Agent._encode_state = spy

def probs(q, ans):
    if q["type"] == "noul":
        return [ans["noul"], 1 - ans["noul"]]
    if q["type"] == "score":
        return [ans["probabilities"][str(i)] for i in range(len(q["criteria"]))]
    return [ans["probabilities"][k] for k in q["criteria"]]

# ---- prompts: every fixture state with every question, one question per call
fx = json.load(open(root / "test/fixtures/prompts.json"))
out = []
for si, s in enumerate(fx["states"]):
    for qi, q in enumerate(fx["questions"]):
        captured.clear()
        try:
            r = a.predict(s, {"q": q})
            it = captured[-1][0]
            out.append({"state": si, "question": qi, "ids": it["ids"], "markers": it["markers"], "qtype": it["qtype"],
                        "probabilities": probs(q, r["answers"]["q"])})
        except Exception as e:  # noqa: BLE001
            out.append({"state": si, "question": qi, "error": f"{type(e).__name__}: {e}"[:300]})
cfg = {"max_len": a.cfg.get("max_len", 512), "head_max_len": a.cfg.get("head_max_len", 192),
       "temperature": a.temperature, "temperature_by_options": a.temperature_by_options}
json.dump({"repo": repo, "subfolder": sub, "config": cfg, "cases": out}, open(root / "test/fixtures/laya-prompts.json", "w"))
print(f"prompts: {sum('ids' in c for c in out)} encoded, {sum('error' in c for c in out)} refused", flush=True)

# ---- Decision Index sample
di = root / "bench/decision-index/sample.json"
if di.exists():
    sample = json.load(open(di))
    res, t0, ok, n = [], time.time(), 0, 0
    for i, r in enumerate(sample):
        ans = a.predict(r["state"], r["questions"])["answers"]
        res.append([probs(q, ans[name]) for name, q in r["questions"].items()])
        for (name, q), p in zip(r["questions"].items(), res[-1]):
            keys = list(q["criteria"]); ok += keys[max(range(len(p)), key=p.__getitem__)] == r["expected"][name]; n += 1
        if i % 40 == 0:
            print(f"{i}/{len(sample)} · {time.time() - t0:.0f} s", flush=True)
    print(f"{repo}: accuracy {ok}/{n} = {ok / n:.3f}")
    json.dump({"repo": repo, "probabilities": res}, open(root / "bench/decision-index/reference-laya.json", "w"))
