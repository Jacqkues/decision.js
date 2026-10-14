// EmbeddingGemma 2 typed decisions: embedding throughput (texts of ?len tokens, as LiteRT-LM's fixed signatures),
// the GPU time per kernel (?prof=1), and where a Decision Index request spends its time (embedding vs head).
import { D1, type Question } from "../../src";
import { Eg2Engine } from "../../src/eg2-engine";
import { tokenize } from "../../src/eg2-prompt";

const out = document.getElementById("out")!;
const log = (s: string) => ((out.textContent += s + "\n"), console.log(s));
const R: Record<string, unknown> = ((window as any).__r = {});
const q = new URLSearchParams(location.search);
const prof = q.get("prof") === "1" ? new Map<string, number>() : undefined;
if (prof) (globalThis as any).__d1Profile = prof;
if (q.get("sk")) { const [tm, tn, ks, below] = q.get("sk")!.split(",").map(Number); (globalThis as any).__d1SplitK = { tm, tn, ks, below }; }
if (q.get("attn")) { const [sq, sg, fq, fg] = q.get("attn")!.split(",").map(Number); (globalThis as any).__eg2Attn = { sliding_attention: { qt: sq, g: sg }, full_attention: { qt: fq, g: fg } }; }
if (q.get("layers")) (globalThis as any).__eg2Layers = Number(q.get("layers"));
if (q.get("tile")) { const [tm, tn, sm, sn, flush] = q.get("tile")!.split(",").map(Number); (globalThis as any).__d1Direct = { tm, tn, sm, sn, flush }; }
try {
  const d1 = await D1.load(`/models/${q.get("model") ?? "eg2-q8"}/`, { matrixUnits: q.get("mu") !== "0", directWeights: q.get("direct") === "0" ? false : undefined });
  const e = d1.engine as Eg2Engine, c = e.m.config;
  const len = Number(q.get("len") ?? 128);
  // texts of exactly `len` tokens
  const base = tokenize(d1.tokenizer, "workflow: support\nsubject: Login broken\nbody: Since the update I can't sign in. ".repeat(40), c.prompt, len);
  for (const n of (q.get("n") ?? "1,8,32,64").split(",").map(Number)) {
    const seqs = Array.from({ length: n }, (_, i) => base.map((id, k) => (k > 2 && k < len - 1 ? ((id + i) % 1000) + 1000 : id)));
    await e.embed(seqs);
    prof?.clear();
    const t = performance.now(), reps = 5;
    for (let r = 0; r < reps; r++) await e.embed(seqs);
    const ms = (performance.now() - t) / reps;
    const g0 = e.stats.gpuMs, t2 = performance.now();
    await e.embed(seqs);
    const all = performance.now() - t2, wait = e.stats.gpuMs - g0;
    log(`${n} texts × ${len} tokens: ${(ms / n).toFixed(2)} ms/text · ${((n * len) / ms).toFixed(0)} tokens/ms · one pass ${all.toFixed(1)} ms = encoding ${(all - wait).toFixed(1)} + GPU ${wait.toFixed(1)}`);
  }
  if (prof) {
    const tot = [...prof.values()].reduce((a, b) => a + b, 0);
    const rows = [...prof].map(([code, ns]) => [(code.match(/\/\/ ([^\n]+)/)?.[1] ?? code.match(/fn (gelu|main)[^\n]*\n[^\n]*\n([^\n]{0,60})/)?.[2] ?? code.slice(0, 60)).replace(/\s+/g, " "), ns] as const);
    log(`  GPU total ${(tot / 1e6 / 5).toFixed(2)} ms per pass (timed one dispatch per pass)`);
    for (const [name, ns] of rows.sort((a, b) => b[1] - a[1]).slice(0, 14)) log(`  ${((100 * ns) / tot).toFixed(1).padStart(5)}%  ${(ns / 1e6 / 5).toFixed(2).padStart(6)} ms  ${name}`);
  }
  {
    // one request as an app sends it: a ticket and 3 questions; the first call embeds the schema, later ones only the state
    const qs = {
      refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
      topic: { type: "choice", instructions: "What is the topic?", criteria: { billing: "billing, payment", delivery: "delivery", bug: "technical bug", account: "login, account" } },
      urgency: { type: "score", instructions: "How urgent is it?", criteria: ["low", "medium", "high"] },
    } as Record<string, Question>;
    const ticket = (i: number) => ({ subject: `Order ${1000 + i}`, body: "I was charged twice for my order and nobody answers my emails. I want a refund now, or I cancel my subscription.", channel: "email", customer: { plan: "pro", tenure_months: 14 + i } });
    const fresh = await D1.load(`/models/${q.get("model") ?? "eg2-q8"}/`, { device: e.device });
    let t = performance.now();
    await fresh.systemOne(ticket(0), qs);
    const first = performance.now() - t;
    const times: number[] = [];
    for (let i = 1; i <= 20; i++) { t = performance.now(); await fresh.systemOne(ticket(i), qs); times.push(performance.now() - t); }
    times.sort((a, b) => a - b);
    const head = performance.now();
    for (let i = 1; i <= 20; i++) await fresh.systemOne(ticket(i), qs); // every text cached: the head alone
    log(`request (state + 3 questions, 11 texts): first ${first.toFixed(1)} ms · then ${times[10]!.toFixed(1)} ms (median, new state each time) · head only ${((performance.now() - head) / 20).toFixed(2)} ms`);
  }
  if (q.get("di") !== "0") {
    const sample = (await (await fetch("/bench/decision-index/sample.json")).json()) as { state: unknown; questions: Record<string, Question> }[];
    const reqs = sample.filter((r) => r.state !== "").map((r) => [r.state, Object.values(r.questions)] as [unknown, Question[]]);
    const fresh = await D1.load(`/models/${q.get("model") ?? "eg2-q8"}/`, { device: e.device });
    const t0 = performance.now(), g0 = (fresh.engine as Eg2Engine).stats.gpuMs;
    await fresh.probabilitiesBatch(reqs);
    const ms = performance.now() - t0, gpu = (fresh.engine as Eg2Engine).stats.gpuMs - g0, st = (fresh.engine as Eg2Engine).stats;
    log(`decision index: ${reqs.length} requests in ${ms.toFixed(0)} ms (${(ms / reqs.length).toFixed(1)} ms/request) · GPU passes ${gpu.toFixed(0)} ms · ${st.tokens} tokens in ${st.forwards} passes`);
    const t1 = performance.now();
    await fresh.probabilitiesBatch(reqs);
    log(`again (every text cached, head only): ${((performance.now() - t1) / reqs.length).toFixed(2)} ms/request`);
  }
  R.done = true;
} catch (err) { log(String((err as Error).stack ?? err)); R.error = String(err); }
