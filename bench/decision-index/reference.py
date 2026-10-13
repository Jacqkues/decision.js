"""Official PyTorch probabilities (original weights, fp32) on bench/decision-index/sample.json.

    python bench/decision-index/reference.py omni|3b   →   bench/decision-index/reference-<model>.json
Per request, each question's probabilities in option order ([yes, no] for a noul), plus the time taken."""
import json, sys, time
from pathlib import Path
import torch
from transformers import AutoModel

name = sys.argv[1]
repo = {"omni": "LiquidAI/d1-omni-600M", "3b": "LiquidAI/d1-3B"}[name]
here = Path(__file__).parent
sample = json.load(open(here / "sample.json"))
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.float32).eval().to(device)
if name == "3b":
    model.engine.device = torch.device(device)  # the runner keeps its own device
run = model.probabilities_batch if name == "omni" else model.engine.probabilities_batch
out, t0 = [], time.time()
with torch.no_grad():
    for i in range(0, len(sample), 8):
        chunk = sample[i:i + 8]
        out += run([(r["state"], list(r["questions"].values())) for r in chunk])
        print(f"{i + len(chunk)}/{len(sample)} · {time.time() - t0:.0f} s", flush=True)
ok = n = 0
for r, probs in zip(sample, out):
    for (qn, q), p in zip(r["questions"].items(), probs):
        keys = list(q["criteria"].keys())
        ok += keys[max(range(len(p)), key=p.__getitem__)] == r["expected"][qn]; n += 1
print(f"{repo}: accuracy {ok}/{n} = {ok / n:.3f}")
json.dump({"repo": repo, "probabilities": out}, open(here / f"reference-{name}.json", "w"))
