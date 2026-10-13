"""A Decision Index sample for the fidelity bench: N requests per benchmark, from a suite rebuilt with the
official kit (github.com/apolinario/decision-index), e.g.

    python -m decision_index suite rebuild --work <work> --only 1 4 12 20 21 27 --skip-download ...
    python bench/decision-index/make-sample.py <work> [N]   →   bench/decision-index/sample.json

The suite is not redistributable (several sources forbid it), so the sample stays out of git."""
import json, random, sys
from pathlib import Path

work, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 60
FAMILIES = ["BFCL-tool-selection", "BANKING77", "ANLI", "ARC-Challenge", "BPoMP-original-limerick", "Humicroedit"]
rng = random.Random(0)
out = []
for fam in FAMILIES:
    rows = [json.loads(l) for l in open(work / "artifacts/benchmark-suite/active/release-v1" / f"{fam}.jsonl")]
    rows = [r for r in rows if r.get("expected") and all(q in r["expected"] for q in r["questions"])]
    for r in rng.sample(rows, min(n, len(rows))):
        out.append({"id": r["id"], "benchmark": fam, "state": r["state"], "questions": r["questions"], "expected": r["expected"]})
    print(f"{fam}: {min(n, len(rows))} of {len(rows)}; questions per request {sum(len(r['questions']) for r in out if r['benchmark'] == fam) / min(n, len(rows)):.1f}")
json.dump(out, open(Path(__file__).parent / "sample.json", "w"), ensure_ascii=False)
