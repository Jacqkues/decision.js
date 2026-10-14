// Runs EmbeddingGemma 2 (LiteRT-LM, ML Drift on WebGPU) and records every WGSL shader it compiles and the
// sequence of dispatches of one embedding call: what ML Drift runs for this architecture.
import { EmbeddingEngine, loadLiteRtLm } from "@litert-lm/core";

const out = document.getElementById("out")!;
const log = (s: string) => (out.textContent += s + "\n");
const r = ((window as any).__r = { done: false, error: "", shaders: [] as { label: string; code: string }[], trace: [] as string[] });
const modules = new Map<GPUShaderModule, number>(), pipes = new Map<GPUComputePipeline, number>();
let tracing = false;
const dev = GPUDevice.prototype as any;
const csm = dev.createShaderModule;
dev.createShaderModule = function (d: GPUShaderModuleDescriptor) {
  const m = csm.call(this, d);
  modules.set(m, r.shaders.length);
  r.shaders.push({ label: d.label ?? "", code: d.code });
  return m;
};
for (const name of ["createComputePipeline", "createComputePipelineAsync"]) {
  const f = dev[name];
  dev[name] = function (d: GPUComputePipelineDescriptor) {
    const res = f.call(this, d);
    const idx = modules.get(d.compute.module) ?? -1;
    if (res instanceof Promise) return res.then((p: GPUComputePipeline) => (pipes.set(p, idx), p));
    pipes.set(res, idx);
    return res;
  };
}
const pass = GPUComputePassEncoder.prototype as any;
let cur = -1;
const sp = pass.setPipeline;
pass.setPipeline = function (p: GPUComputePipeline) { cur = pipes.get(p) ?? -2; return sp.call(this, p); };
const dw = pass.dispatchWorkgroups;
pass.dispatchWorkgroups = function (x: number, y = 1, z = 1) { if (tracing) r.trace.push(`${cur} ${x}x${y}x${z}`); return dw.call(this, x, y, z); };

try {
  const q = new URLSearchParams(location.search);
  const len = Number(q.get("len") ?? 128);
  await loadLiteRtLm("/node_modules/@litert-lm/core/wasm");
  let t = performance.now();
  const engine = await EmbeddingEngine.create({ model: "/compare/litert-lm/embeddinggemma-2-text-270m.litertlm", maxInputLength: len, minInputLength: len });
  log(`loaded in ${(performance.now() - t).toFixed(0)} ms, ${r.shaders.length} shaders`);
  const text = "task: classification | query: workflow: support\nsubject: Login broken\nbody: Since the update I can't sign in.";
  const e0 = await engine.computeEmbedding(text);
  log(`dim ${e0.embedding.length}, input ${e0.inputLength} tokens, first ${e0.embedding.slice(0, 4).map((v) => v.toFixed(4))}`);
  tracing = true;
  await engine.computeEmbedding(text);
  tracing = false;
  log(`one call: ${r.trace.length} dispatches, ${new Set(r.trace.map((s) => s.split(" ")[0])).size} distinct shaders`);
  for (const n of [1, 8, 32]) {
    const batch = Array.from({ length: n }, (_, i) => `${text} ${i}`);
    await engine.computeEmbeddingBatch(batch);
    t = performance.now();
    for (let k = 0; k < 3; k++) await engine.computeEmbeddingBatch(batch);
    log(`batch ${n}: ${((performance.now() - t) / 3 / n).toFixed(2)} ms/text`);
  }
  log(`${r.shaders.length} shaders in total`);
  r.done = true;
} catch (e) {
  r.error = String((e as Error).stack ?? e);
  log("error: " + r.error);
}
