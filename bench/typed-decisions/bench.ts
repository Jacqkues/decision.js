// decision.js on the test split of LocalLLaMA/typed-decisions (400 cases × 5 questions, soft gold labels from
// several annotators), read from the Hugging Face datasets server: ?model=<dir under /models/>.
// Accuracy is argmax against the gold label; KL is KL(gold ‖ model); Brier is the sum over options of
// (p − gold)², averaged over decisions (the dataset card's definitions: uniform scores 0.444 and 0.238).
import { D1, type Question } from "../../src";

interface Row { id: string; workflow: string; state: unknown; questions: Record<string, Question>; gold: Record<string, { label: string; probabilities: Record<string, number> }> }
const out = document.getElementById("out")!;
const log = (s: string) => ((out.textContent += s + "\n"), console.log(s));
const R: Record<string, unknown> = ((window as any).__r = {});
const q = new URLSearchParams(location.search);

/** The option names of a question, in the order decision.js returns its probabilities. */
const keys = (qq: Question) => (qq.type === "noul" ? ["true", "false"] : qq.type === "score" ? qq.criteria.map((_, i) => String(i)) : Object.keys(qq.criteria));

try {
  const rows: Row[] = [];
  for (let offset = 0; offset < 400; offset += 100) {
    const res = await fetch(`https://datasets-server.huggingface.co/rows?dataset=LocalLLaMA/typed-decisions&config=all&split=test&offset=${offset}&length=100`);
    if (!res.ok) throw new Error(`datasets server: HTTP ${res.status}`);
    for (const { row } of (await res.json()).rows as { row: Record<string, string> }[])
      rows.push({ id: row.id!, workflow: row.workflow!, state: JSON.parse(row.state!), questions: JSON.parse(row.questions!), gold: JSON.parse(row.gold!) });
  }
  const model = q.get("model") ?? "eg2-q8";
  const d1 = await D1.load(`/models/${model}/`, { matrixUnits: q.get("mu") !== "0" });
  const preds: number[][][] = new Array(rows.length);
  const t = performance.now();
  // one batch per workflow: EmbeddingGemma 2 names the workflow in the state's text (the other models ignore it)
  for (const wf of [...new Set(rows.map((r) => r.workflow))]) {
    const idx = rows.map((r, i) => (r.workflow === wf ? i : -1)).filter((i) => i >= 0);
    d1.workflow = q.get("workflow") === "0" ? "" : wf; // ?workflow=0: without the workflow name
    const got = await d1.probabilitiesBatch(idx.map((i) => [rows[i]!.state, Object.values(rows[i]!.questions)]));
    idx.forEach((i, k) => (preds[i] = got[k]!));
  }
  const ms = performance.now() - t;
  type Acc = { n: number; ok: number; kl: number; brier: number; uKl: number; uBrier: number };
  const acc = new Map<string, Acc>();
  const add = (key: string, ok: boolean, kl: number, brier: number, uKl: number, uBrier: number) => {
    const a = acc.get(key) ?? (acc.set(key, { n: 0, ok: 0, kl: 0, brier: 0, uKl: 0, uBrier: 0 }), acc.get(key)!);
    a.n++; a.ok += +ok; a.kl += kl; a.brier += brier; a.uKl += uKl; a.uBrier += uBrier;
  };
  rows.forEach((r, i) => Object.entries(r.questions).forEach(([name, qq], j) => {
    const ks = keys(qq), p = preds[i]![j]!, g = r.gold[name]!;
    const gp = ks.map((k) => g.probabilities[k] ?? 0);
    const best = ks[p.indexOf(Math.max(...p))]!;
    const kl = (pp: number[]) => gp.reduce((s, x, k) => s + (x > 0 ? x * Math.log(x / Math.max(pp[k]!, 1e-12)) : 0), 0);
    const br = (pp: number[]) => gp.reduce((s, x, k) => s + (pp[k]! - x) ** 2, 0);
    const u = ks.map(() => 1 / ks.length);
    for (const key of ["all", qq.type, r.workflow]) add(key, best === g.label, kl(p), br(p), kl(u), br(u));
  }));
  log(`${d1.kind} (${model}) · LocalLLaMA/typed-decisions test · ${rows.length} cases · ${(ms / rows.length).toFixed(1)} ms per case`);
  log(`${"".padEnd(28)} ${"n".padStart(5)}  accuracy  KL      Brier   (uniform: KL, Brier)`);
  for (const [k, a] of acc) log(`${k.padEnd(28)} ${String(a.n).padStart(5)}  ${(a.ok / a.n).toFixed(3)}     ${(a.kl / a.n).toFixed(3)}   ${(a.brier / a.n).toFixed(3)}   (${(a.uKl / a.n).toFixed(3)}, ${(a.uBrier / a.n).toFixed(3)})`);
  Object.assign(R, { done: true, accuracy: acc.get("all")!.ok / acc.get("all")!.n });
} catch (e) { log(String((e as Error).stack ?? e)); R.error = String(e); }
