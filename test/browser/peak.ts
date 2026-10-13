// The matrix-unit ceiling of this GPU through WebGPU: subgroup-matrix multiply-accumulates on registers only
// (no memory traffic), f16 operands with f16 or f32 accumulation, timed over many workgroups.
import { requestDevice } from "../../src";
const out = document.getElementById("out")!;
const R: Record<string, unknown> = ((window as any).__r = {});
const log = (s: string) => ((out.textContent += s + "\n"), console.log(s));
try {
  const device = await requestDevice();
  const q = new URLSearchParams(location.search); const ITER = 2048, ACC = Number(q.get("acc") ?? "8"), WG = 4096;
  for (const acc of ["f16", "f32"]) {
    const code = `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);
@group(0) @binding(0) var<storage, read_write> o: array<${acc}>;
var<workgroup> scratch: array<${acc}, ${64 * ACC * 4}>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let sub = li / 32u;
  let a = subgroup_matrix_left<f16, 8, 8>(f16(0.001) * f16(f32(wg.x % 7u)));
  let b = subgroup_matrix_right<f16, 8, 8>(f16(0.002));
${Array.from({ length: ACC }, (_, i) => `  var c${i} = subgroup_matrix_result<${acc}, 8, 8>();`).join("\n")}
  for (var i = 0u; i < ${ITER}u; i++) {
${Array.from({ length: ACC }, (_, i) => `    c${i} = subgroupMatrixMultiplyAccumulate(a, b, c${i});`).join("\n")}
  }
${Array.from({ length: ACC }, (_, i) => `  subgroupMatrixStore<row_major>(&scratch, sub * ${64 * ACC}u + ${i * 64}u, c${i}, 8u);`).join("\n")}
  workgroupBarrier();
  if (li == 0u) { o[wg.x] = scratch[0] + scratch[${64 * ACC * 4 - 1}]; }
}`;
    const module = device.createShaderModule({ code });
    const pipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const buf = device.createBuffer({ size: WG * 4, usage: GPUBufferUsage.STORAGE });
    const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
    const run = async () => {
      const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
      pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(WG); pass.end();
      const t = performance.now(); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); return performance.now() - t;
    };
    await run();
    let best = Infinity; for (let r = 0; r < 5; r++) best = Math.min(best, await run());
    const flops = WG * 4 /* subgroups */ * ITER * ACC * 2 * 8 * 8 * 8;
    log(`MMA ${acc} accumulate, ${ACC} accumulators: ${(flops / best / 1e9).toFixed(2)} TFLOP/s (${best.toFixed(1)} ms)`);
  }
  R.done = true;
} catch (e) { log(String((e as Error).stack)); R.error = String(e); }
