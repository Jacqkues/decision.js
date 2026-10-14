"""References from the EmbeddingGemma 2 typed-decision model's own code (fp32, transformers==5.19.0):

    python scripts/eg2-reference.py
      → test/fixtures/eg2-prompts.json: the rendered text and token ids of every state of test/fixtures/prompts.json
        (with and without a workflow) and of every question and option text (or the model's error)
      → bench/decision-index/reference-eg2.json: probabilities on the Decision Index sample, if present
Probabilities are in option order ([yes, no] for a noul, as d1.js returns them)."""
import json, time
from pathlib import Path
import torch
from transformers import AutoModel

repo = "mrm8488/embeddinggemma-2-ft-typed-decision"
root = Path(__file__).resolve().parent.parent
model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.float32).eval()
mod = __import__(type(model).__module__, fromlist=["render_state"])
tok = model.get_tokenizer()


def ids(text):
    return model.tokenize([text])["input_ids"][0].tolist()


def probs(q, ans):
    p = ans.probabilities
    if q["type"] == "noul":
        return [p["true"], p["false"]]
    if q["type"] == "score":
        return [p[str(i)] for i in range(len(q["criteria"]))]
    return [p[k] for k in q["criteria"]]


fx = json.load(open(root / "test/fixtures/prompts.json"))
states = []
for si, s in enumerate(fx["states"]):
    for wf in ("", "agent_trace_observability"):
        try:
            text = mod.render_state(wf, s)
            states.append({"state": si, "workflow": wf, "text": text, "ids": ids(text)})
        except Exception as e:  # noqa: BLE001
            states.append({"state": si, "workflow": wf, "error": f"{type(e).__name__}: {e}"[:300]})
questions = []
for qi, q in enumerate(fx["questions"]):
    spec = mod.parse_question("q", q)
    texts = [mod.render_question(spec)] + [mod.render_option(spec, i) for i in range(len(spec.options))]
    questions.append({"question": qi, "texts": texts, "ids": [ids(t) for t in texts], "options": list(spec.options)})
json.dump({"repo": repo, "prompt": model.config.prompt, "states": states, "questions": questions}, open(root / "test/fixtures/eg2-prompts.json", "w"))
print(f"prompts: {sum('ids' in s for s in states)} states rendered, {sum('error' in s for s in states)} refused, {len(questions)} questions", flush=True)

di = root / "bench/decision-index/sample.json"
if di.exists():
    sample = json.load(open(di))
    res, t0, ok, n = [], time.time(), 0, 0
    for r in sample:
        try:
            ans = model.decide(r["state"], r["questions"])
        except Exception:  # noqa: BLE001 (an empty string state)
            res.append(None)
            continue
        res.append([probs(q, ans[name]) for name, q in r["questions"].items()])
        for (name, q), p in zip(r["questions"].items(), res[-1]):
            keys = ["yes", "no"] if q["type"] == "noul" else list(q["criteria"])
            ok += keys[max(range(len(p)), key=p.__getitem__)] == r["expected"][name]; n += 1
    print(f"decision index: accuracy {ok}/{n} = {ok / n:.3f}, {len(res)} requests ({res.count(None)} refused) in {time.time() - t0:.0f} s", flush=True)
    json.dump({"repo": repo, "probabilities": res}, open(root / "bench/decision-index/reference-eg2.json", "w"))
