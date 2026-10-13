// d1.js against LiteRT.js (ML Drift on WebGPU) on the same 300 tickets × 3 questions (900 sequences,
// compare/litert/sequences.json), both against the official PyTorch probabilities (reference.json).
//   ?tflite=d1-omni-b24-l136.tflite &prec=fp16|fp32 &d1=d1-omni-600M-q8 (a dir under /models/, "" to skip)
import { loadAndCompile, loadLiteRt, Tensor } from "@litertjs/core";
import { D1OmniEngine, temperatureKey, type OmniSeq, type Question } from "../../src";

interface Seq { ids: number[]; markers: number[]; type: Question["type"] }
const out = document.getElementById("out")!;
const log = (s: string) => ((out.textContent += s + "\n"), console.log(s));
const R: Record<string, unknown> = {};
(window as unknown as { __r: unknown }).__r = R;
const q = new URLSearchParams(location.search);
const argmax = (p: number[]) => p.indexOf(Math.max(...p));
const softmax = (z: number[]) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b); return e.map((v) => v / s); };

const { questions, sequences } = (await (await fetch("/compare/litert/sequences.json")).json()) as { questions: Question[]; sequences: Seq[] };
const ref = (await (await fetch("/compare/litert/reference.json")).json()) as { probabilities: number[][]; temperatures: Record<string, number> };
const T = sequences.length / 3; // tickets
const probs = (z: number[], i: number) => {
  const qq = questions[i % 3]!, t = ref.temperatures[temperatureKey(qq)] ?? 1, p = softmax(z.map((v) => v / t));
  return qq.type === "noul" ? p.reverse() : p;
};
const score = (name: string, got: number[][], ms: number) => {
  let worst = 0, same = 0;
  got.forEach((p, i) => { worst = Math.max(worst, ...p.map((v, k) => Math.abs(v - ref.probabilities[i]![k]!))); same += +(argmax(p) === argmax(ref.probabilities[i]!)); });
  log(`${name.padEnd(34)} ${(ms / T).toFixed(1).padStart(6)} ms/ticket · same answer as PyTorch fp32 ${same}/${got.length} · largest probability gap ${worst.toFixed(4)}`);
  R[name] = { msPerTicket: ms / T, same, worst };
};

try {
  // ---- d1.js
  const d1dir = q.get("d1") ?? "d1-omni-600M-q8";
  if (d1dir) {
    const engine = await D1OmniEngine.load(`/models/${d1dir}/`, { matrixUnits: q.get("mu") !== "0" });
    const seqs: OmniSeq[] = sequences.map((s) => ({ ids: s.ids, markers: s.markers, type: s.type }));
    await engine.logits(seqs.slice(0, 24));
    const t = performance.now(), got: number[][] = [];
    for (let i = 0; i < seqs.length; i += 72) (await engine.logits(seqs.slice(i, i + 72))).logits.forEach((z) => got.push(probs(z, got.length)));
    score(`d1.js ${d1dir} (${engine.matrixUnits ? "matrix units" : "portable"})`, got, performance.now() - t);
    engine.destroy();
  }
  // ---- LiteRT.js
  const file = q.get("tflite") ?? "d1-omni-b24-l136.tflite";
  const [, B, L] = file.match(/-b(\d+)-l(\d+)/)!.map(Number) as [number, number, number];
  const prec = (q.get("prec") ?? "fp16") as "fp16" | "fp32";
  await loadLiteRt("/node_modules/@litertjs/core/wasm/");
  let t = performance.now();
  // ?parts=N: the model is N chained graphs (…-partIofN.tflite), the hidden states staying on the GPU
  const parts = Number(q.get("parts") ?? "1");
  const files = parts > 1 ? Array.from({ length: parts }, (_, i) => file.replace(".tflite", `-part${i}of${parts}.tflite`)) : [file];
  const models = [];
  for (const f of files) models.push(await loadAndCompile(`/compare/litert/${f}`, { accelerator: "webgpu", gpuOptions: { precision: prec } }));
  log(`LiteRT.js compiled ${files.length} graph(s) in ${((performance.now() - t) / 1000).toFixed(1)} s · fully on the GPU: ${models.every((m) => m.isFullyAccelerated)}`);
  const QT = { choice: 0, score: 1, noul: 2 };
  const embed = new Uint16Array(await (await fetch("/compare/litert/embed.f16")).arrayBuffer()), D = 1024;
  const f16 = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) { const s = i & 0x8000 ? -1 : 1, e = (i >> 10) & 31, f = i & 1023; f16[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? NaN : s * (1 + f / 1024) * 2 ** (e - 15); }
  const inputs = (chunk: Seq[]) => {
    const h = new Float32Array(B * L * D), pad = new Float32Array(B * L), qt = new Float32Array(B * 3);
    chunk.forEach((s, i) => {
      s.ids.forEach((id, t) => { const o = ((i * L + t) * D); for (let d = 0; d < D; d++) h[o + d] = f16[embed[id * D + d]!]!; });
      pad.fill(1, i * L, i * L + s.ids.length);
      qt[i * 3 + QT[s.type]] = 1;
    });
    return [new Tensor(h, [B, L, D]), new Tensor(pad, [B, L]), new Tensor(qt, [B, 3])];
  };
  let prepMs = 0;
  const run = async (chunk: Seq[]) => {
    const t0 = performance.now();
    const ins = inputs(chunk);
    prepMs += performance.now() - t0;
    let h = ins[0]!;
    const temps: { delete(): void }[] = [...ins];
    for (let i = 0; i < models.length - 1; i++) {
      h = (await models[i]!.run([h, ins[1]!]))[0]!;
      temps.push(h);
      if (q.get("debug") && !(globalThis as any).__shown) {
        const v = (await h.data()) as Float32Array;
        log(`  after part ${i}: mean |h| ${(v.reduce((a, b) => a + Math.abs(b), 0) / v.length).toFixed(3)}, first ${Array.from(v.subarray(0, 4)).map((x) => x.toFixed(3)).join(" ")}`);
      }
    }
    const res = await models[models.length - 1]!.run([h, ...ins.slice(1)]);
    const z = (await res[0]!.data()) as Float32Array;
    if (q.get("debug") && !(globalThis as any).__shown) {
      (globalThis as any).__shown = 1;
      log(`  raw logits of seq 0: ${Array.from(z.subarray(0, 7)).map((v) => v.toFixed(3)).join(" ")} · output on ${res[0]!.accelerator}`);
      const hh = (await h.data()) as Float32Array; // the hidden state entering the last graph
      log(`  last graph input h: ${hh.length} values, mean |h| ${(hh.reduce((a, b) => a + Math.abs(b), 0) / hh.length).toFixed(3)}, first ${Array.from(hh.subarray(0, 4)).map((v) => v.toFixed(3)).join(" ")}`);
      const h0 = (await ins[0]!.data()) as Float32Array;
      log(`  embeddings in: mean |h| ${(h0.reduce((a, b) => a + Math.abs(b), 0) / h0.length).toFixed(4)}`);
    }
    temps.forEach((x) => x.delete()); res.forEach((x) => x.delete());
    return chunk.map((s, i) => s.markers.map((m) => z[i * L + m]!));
  };
  await run(sequences.slice(0, B));
  t = performance.now();
  const got: number[][] = [];
  for (let i = 0; i < sequences.length; i += B) (await run(sequences.slice(i, i + B))).forEach((z) => got.push(probs(z, got.length)));
  if (q.get("debug")) for (const i of [0, 1, 2, 3, 4, 5]) log(`  seq ${i}: LiteRT ${got[i]!.map((v) => v.toFixed(3)).join(" ")} · PyTorch ${ref.probabilities[i]!.map((v) => v.toFixed(3)).join(" ")}`);
  if (q.get("debug")) log(`  input details of each graph: ${models.map((m) => m.getInputDetails().map((d) => `${d.name}${JSON.stringify(Object.values(d.shape))}`).join(",")).join(" | ")}`);
  log(`  (of which ${(prepMs / T).toFixed(1)} ms/ticket building the inputs in JS: embedding rows, padding)`);
  score(`LiteRT.js / ML Drift (${prec}, batch ${B}, padded to ${L})`, got, performance.now() - t);
  R.done = true;
} catch (e) {
  log(`error: ${(e as Error).stack ?? e}`);
  R.error = String(e);
}
