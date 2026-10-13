// Throughput of systemOneBatch: ?model=<dir under /models/> &n=<tickets> &mu=0 (portable kernels) &reps=<runs, best kept>.
// Tickets are made up from a few templates (about 80 to 120 words each); 3 questions per ticket.
import { D1 } from "../../src";
import * as K from "../../src/kernels";
import * as K3 from "../../src/d3-kernels";
import { questions, tickets as makeTickets } from "./tickets";

const out = document.getElementById("out")!;
const log = (s: string) => ((out.textContent += s + "\n"), console.log(s));
out.textContent = "";
const R: Record<string, unknown> = {};
(window as unknown as { __r: unknown }).__r = R;
const q = new URLSearchParams(location.search);
const N = Number(q.get("n") ?? "300");
const tickets = makeTickets(N);

try {
  if (q.get("gdnform")) (globalThis as any).__d3GdnForm = q.get("gdnform");
  const ab = q.get("gdnab"); // profiling: the d3 DeltaNet kernel without one of its steps (prep, rec, norm)
  if (ab) (globalThis as any).__d3GdnPatch = (code: string) => {
    const cut = (from: string, to: string) => { // never runs, but keeps the bindings it uses
      const i = code.indexOf(from), j = code.indexOf(to);
      return code.slice(0, i) + "if (p.eps < 0.0) {\n    " + code.slice(i, j) + "}\n    " + code.slice(j);
    };
    if (ab === "prep") return cut("// 1. conv, norms and gates", "// 2. the recurrence");
    if (ab === "rec") return cut("// 2. the recurrence", "// 3. gated RMSNorm");
    return cut("// 3. gated RMSNorm", "\n  }\n}");
  };
  if (q.get("skip")) (globalThis as any).__d1Skip = [q.get("skip")]; // profiling: drop one kind of kernel
  const dk = q.get("dk"); // direct kernel experiment: tm,tn,sm,sn[,flush]
  if (dk) { const [tm, tn, sm, sn, flush] = dk.split(",").map(Number) as number[]; (globalThis as any).__d1Direct = { tm, tn, sm, sn, flush }; }
  const pf = q.get("pf"); // portable f16 kernel experiment: tm,tn,rm,rn,aShared(0|1)
  if (pf) { const [tm, tn, rm, rn, a] = pf.split(",").map(Number) as number[]; (globalThis as any).__d1PortableF16 = { tm, tn, rm, rn, aShared: a === 1 }; }
  const pk = q.get("pk"); // portable kernel experiment: tm,tn,rm,rn[,f16]
  if (pk) { const [tm, tn, rm, rn] = pk.split(",").map(Number) as number[]; (globalThis as any).__d1Portable = { tm, tn, rm, rn }; }
  if (q.get("lanes")) (globalThis as any).__d3Lanes = Number(q.get("lanes")); // d3 DeltaNet kernel experiment: lanes per row
  const at = q.get("attn"); // d3 attention experiment: queries,heads per workgroup
  if (at) { const [qt, g] = at.split(",").map(Number); (globalThis as any).__d3Attn = { qt, g }; }
  if (q.get("prof")) (globalThis as any).__d1Profile = new Map<string, number>(); // GPU time per kernel
  const d1 = await D1.load(`/models/${q.get("model") ?? "d1-omni-600M-q8"}/`, { shareState: q.get("share") === "1", matrixUnits: q.get("mu") !== "0", directWeights: q.get("direct") === "0" ? false : undefined });
  log(`${d1.kind} · matrix units ${d1.engine.matrixUnits} · ${N} tickets × 3 questions`);
  await d1.systemOneBatch(tickets.slice(0, 8).map((t) => [t, questions])); // warm-up
  let ms = Infinity, res: Awaited<ReturnType<typeof d1.systemOneBatch>> = [];
  for (let r = 0; r < Number(q.get("reps") ?? "1"); r++) { // &reps=n: the fastest of n runs
    const t = performance.now();
    res = await d1.systemOneBatch(tickets.map((t) => [t, questions]));
    ms = Math.min(ms, (performance.now() - t) / N);
  }
  const tokens = res.reduce((a, r) => a + r.usage.input_tokens, 0) / N;
  log(`${ms.toFixed(1)} ms per ticket (${tokens.toFixed(0)} tokens read per ticket) → ${((ms * 1000) / 1000).toFixed(0)} s per 1,000 tickets`);
  const prof = (globalThis as any).__d1Profile as Map<string, number> | undefined;
  if (prof) {
    prof.clear(); // only the last run: one more, timed per kernel
    await d1.systemOneBatch(tickets.map((t) => [t, questions]));
    const names = new Map<string, string>();
    for (const [k, v] of Object.entries({ ...K, ...K3 })) if (typeof v === "string") names.set(v, k);
    for (const qt of [4, 8, 16]) for (const g of [1, 2, 4]) names.set(K3.attention256(qt, g), `attention256(${qt}, ${g})`);
    for (const l of [2, 4] as const) names.set(K3.gdnLayer(l), `gdnLayer(${l})`);
    const total = [...prof.values()].reduce((a, b) => a + b, 0);
    log(`GPU time per ticket, kernel by kernel (each in its own pass): ${(total / 1e6 / N).toFixed(1)} ms`);
    for (const [code, ns] of [...prof].sort((a, b) => b[1] - a[1]))
      log(`  ${(names.get(code) ?? code.match(/\/\/ direct tile \S+|binding\(1\)[^;]*/)?.[0] ?? "?").padEnd(28)} ${(ns / 1e6 / N).toFixed(2).padStart(6)} ms  ${((100 * ns) / total).toFixed(1)}%`);
  }
  Object.assign(R, { done: true, ms });
} catch (e) {
  log(`error: ${(e as Error).stack ?? e}`);
  Object.assign(R, { error: String(e) });
}
