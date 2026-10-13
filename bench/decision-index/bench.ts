// d1.js against the official PyTorch model (original weights) on a Decision Index sample
// (make-sample.py, reference.py): ?model=<dir under /models/> &ref=omni|3b &mu=0
import { D1, type Question } from "../../src";

interface Row { id: string; benchmark: string; state: unknown; questions: Record<string, Question>; expected: Record<string, string> }
const out = document.getElementById("out")!;
const log = (s: string) => ((out.textContent += s + "\n"), console.log(s));
const R: Record<string, unknown> = {};
(window as unknown as { __r: unknown }).__r = R;
const q = new URLSearchParams(location.search);
const argmax = (p: number[]) => p.indexOf(Math.max(...p));

try {
  const sample = (await (await fetch("/bench/decision-index/sample.json")).json()) as Row[];
  const ref = (await (await fetch(`/bench/decision-index/reference-${q.get("ref") ?? "omni"}.json`)).json()) as { repo: string; probabilities: number[][][] };
  const d1 = await D1.load(`/models/${q.get("model") ?? "d1-omni-600M-q8"}/`, { shareState: q.get("share") === "1", matrixUnits: q.get("mu") !== "0" });
  log(`${d1.kind} (${q.get("model") ?? "d1-omni-600M-q8"}) vs ${ref.repo} (official runtime, original weights) · ${sample.length} Decision Index requests · matrix units ${d1.engine.matrixUnits}`);
  const requests = sample.map((r) => [r.state, Object.values(r.questions)] as [unknown, Question[]]);
  await d1.probabilitiesBatch(requests.slice(0, 4));
  const t = performance.now();
  const got = await d1.probabilitiesBatch(requests);
  const ms = performance.now() - t;
  type Acc = { n: number; ok: number; okRef: number; same: number; worst: number };
  const by = new Map<string, Acc>(), all: Acc = { n: 0, ok: 0, okRef: 0, same: 0, worst: 0 };
  sample.forEach((r, i) => Object.entries(r.questions).forEach(([name, qq], j) => {
    const keys = Object.keys((qq as { criteria: object }).criteria), p = got[i]![j]!, pr = ref.probabilities[i]![j]!;
    for (const a of [by.get(r.benchmark) ?? (by.set(r.benchmark, { n: 0, ok: 0, okRef: 0, same: 0, worst: 0 }), by.get(r.benchmark)!), all]) {
      a.n++;
      a.ok += +(keys[argmax(p)] === r.expected[name]);
      a.okRef += +(keys[argmax(pr)] === r.expected[name]);
      a.same += +(argmax(p) === argmax(pr));
      a.worst = Math.max(a.worst, ...p.map((v, k) => Math.abs(v - pr[k]!)));
    }
  }));
  const pct = (a: number, n: number) => `${((100 * a) / n).toFixed(1)}%`.padStart(6);
  log(`\n${"benchmark".padEnd(26)} ${"questions".padStart(9)}  decision.js acc  official acc  same answer  largest gap`);
  for (const [b, a] of [...by, ["all", all] as const]) log(`${b.padEnd(26)} ${String(a.n).padStart(9)}  ${pct(a.ok, a.n)}     ${pct(a.okRef, a.n)}       ${pct(a.same, a.n)}       ${a.worst.toFixed(3)}`);
  log(`\n${(ms / sample.length).toFixed(1)} ms per request (${sample.length} requests, ${all.n} questions)`);
  Object.assign(R, { done: true, all, ms });
} catch (e) {
  log(`error: ${(e as Error).stack ?? e}`);
  R.error = String(e);
}
