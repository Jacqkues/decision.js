// d1.js against LiquidAI's PyTorch code, through the public API: ?model=<dir under /models/>.
// Every request of test/fixtures/prompts.json (each state with every question); reference.json holds the
// official probabilities with the original weights and with the exported (quantized) ones.
import { D1, type Question } from "../../src";
import fx from "../fixtures/prompts.json";

const out = document.getElementById("out")!;
const lines: string[] = [];
const log = (s: string) => (lines.push(s), (out.textContent = lines.join("\n")), console.log(s));
const R: Record<string, unknown> = {};
(window as unknown as { __r: unknown }).__r = R;
const q = new URLSearchParams(location.search);
const base = `/models/${q.get("model") ?? "d1-omni-600M-q8"}/`;
if (q.get("lanes")) (globalThis as any).__d3Lanes = Number(q.get("lanes")); // d3: force the 2-lane DeltaNet kernel
const argmax = (p: number[]) => p.indexOf(Math.max(...p));

try {
  const pf = q.get("pf");
  if (pf) { const [tm, tn, rm, rn, a] = pf.split(",").map(Number) as number[]; (globalThis as any).__d1PortableF16 = { tm, tn, rm, rn, aShared: a === 1 }; }
  const pk = q.get("pk");
  if (pk) { const [tm, tn, rm, rn] = pk.split(",").map(Number) as number[]; (globalThis as any).__d1Portable = { tm, tn, rm, rn }; }
  if (q.get("d3lanes")) (globalThis as any).__d3Lanes = Number(q.get("d3lanes"));
  if (q.get("gdnform")) (globalThis as any).__d3GdnForm = q.get("gdnform");
  let t = performance.now();
  const d1 = await D1.load(base, { matrixUnits: q.get("mu") !== "0", directWeights: q.get("direct") === "0" ? false : undefined });
  log(`${d1.kind} loaded in ${((performance.now() - t) / 1000).toFixed(1)} s · matrix units ${d1.engine.matrixUnits} · subgroups ${d1.engine.subgroups}`);
  const ref = (await (await fetch(`${base}reference.json`)).json()) as { original: (number[] | null)[][]; quantized: (number[] | null)[][] };
  // the questions the reference runtime answered (null: it refused them, and d1.js refuses them too)
  const kept = (fx.states as unknown[]).map((_, i) => (fx.questions as Question[]).map((_, j) => j).filter((j) => ref.quantized[i]![j] != null)).map((js, i) => ({ i, js })).filter((r) => r.js.length);
  const requests = kept.map(({ i, js }) => [(fx.states as unknown[])[i], js.map((j) => (fx.questions as Question[])[j]!)] as [unknown, Question[]]);
  await d1.probabilitiesBatch(requests.slice(0, 1)); // warm-up
  t = performance.now();
  const got = await d1.probabilitiesBatch(requests);
  const ms = performance.now() - t;
  let worst = 0, sameQ = 0, sameO = 0, n = 0;
  got.forEach((perQ, r) => perQ.forEach((p, jj) => {
    const i = kept[r]!.i, j = kept[r]!.js[jj]!;
    const rq = ref.quantized[i]![j]!, ro = ref.original[i]![j]!;
    const gap = Math.max(...p.map((v, k) => Math.abs(v - rq[k]!)));
    if (gap > 0.05 && q.get("debug")) log(`  state ${i} question ${j}: gap ${gap.toFixed(3)} · d1.js ${p.map((v) => v.toFixed(3)).join(" ")} · PyTorch ${rq.map((v) => v.toFixed(3)).join(" ")}`);
    worst = Math.max(worst, gap);
    sameQ += +(argmax(p) === argmax(rq));
    sameO += +(argmax(p) === argmax(ro));
    n++;
  }));
  log(`${n} decisions in ${ms.toFixed(0)} ms · largest probability difference vs PyTorch (same weights) ${worst.toFixed(4)}`);
  log(`same answer as PyTorch with the same weights ${sameQ}/${n} · as the original model ${sameO}/${n}`);
  const r = await d1.systemOne("I was charged twice for my order and nobody answers my emails. I want a refund now.", {
    refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
    topic: { type: "choice", instructions: "What is the topic?", criteria: { billing: "billing, payment", delivery: "delivery", bug: "technical bug" } },
    urgency: { type: "score", instructions: "How urgent is it?", criteria: ["low", "medium", "high"] },
  });
  log(`systemOne: ${JSON.stringify(r)}`);
  Object.assign(R, { done: true, worst, sameQ, sameO, n });
} catch (e) {
  log(`error: ${(e as Error).stack ?? e}`);
  Object.assign(R, { error: String(e) });
}
