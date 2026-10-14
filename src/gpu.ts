// What the engines share: a WebGPU device with f16 (and matrix units when the browser exposes them),
// weights streamed from one weights.bin by Range requests (manifest.json says where each tensor lives),
// cached pipelines, and a recorder that encodes a whole forward pass into one command buffer.
import { DIRECT_TILE, MATMUL_DIRECT, MATMUL_SWIGLU, matmulDirect, matmulSplitK, MATMUL_PORTABLE_F16, MATMUL_PORTABLE_Q4, MATMUL_PORTABLE_Q8, MATMUL_Q4_SG32, MATMUL_Q8_SG32, matmulPortable, matmulPortableF16, PORTABLE_F16_TILE, SILU_MUL, tileWeights, tileWeightsPortable } from "./kernels.js";

export interface TensorInfo { dtype: "q4" | "q8" | "f16" | "f32"; shape: number[]; offset: number; bytes: number; scales?: number; scales_bytes?: number }
export interface BaseManifest { tensors: Record<string, TensorInfo>; bytes: number }
/** A tensor on the GPU: quantized (codes + scales), f16 (data), or, once converted for the matmul, f16 in
 *  8×8 blocks (tiled, matrix units) or k-major in groups of 4 columns (cols4, portable). swiglu: a tiled fused
 *  w1 | w3 reordered for MATMUL_SWIGLU. */
export interface Weight { codes?: GPUBuffer; scales?: GPUBuffer; data?: GPUBuffer; tiled?: GPUBuffer; cols4?: GPUBuffer; shape: number[]; bits?: 4 | 8; swiglu?: boolean }

export interface GpuOptions {
  /** Use the GPU's matrix units when available (default true); false forces the portable kernels. */
  matrixUnits?: boolean;
  /** Convert the quantized linear weights once to f16 (8×8 blocks on matrix units, k-major groups of 4
   *  columns otherwise), which the matmul reads with no dequantization (faster; twice the GPU memory of q8).
   *  Default: when that stays under 1.5 GB, and not on a phone or tablet, whose GPU shares a few GB with
   *  everything else. */
  directWeights?: boolean;
  /** Called as weights arrive, in bytes. */
  onProgress?: (loaded: number, total: number) => void;
  /** An existing device (it needs shader-f16, and subgroups / subgroup matrices for the fast kernels). */
  device?: GPUDevice;
  /** fetch to use (e.g. one that caches); defaults to globalThis.fetch. */
  fetch?: typeof fetch;
}

/** Profiling: set globalThis.__d1Profile to a Map before loading, and each forward pass adds the GPU time of
 *  every kernel to it (ns, keyed by WGSL source). Each dispatch then gets its own timed pass (slower). */
/** A phone or tablet (Chromium says so directly; elsewhere the user agent does). */
const mobile = () => (navigator as { userAgentData?: { mobile?: boolean } }).userAgentData?.mobile || /Android|iPhone|iPad|Mobile/.test(navigator.userAgent);

const profiling = () => (globalThis as { __d1Profile?: Map<string, number> }).__d1Profile;

export const align4 = (n: number) => Math.ceil(n / 4) * 4;

export async function requestDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("decision.js: WebGPU is not available");
  if (!adapter.features.has("shader-f16")) throw new Error("decision.js: this GPU has no shader-f16");
  const wanted = ["subgroups", "chromium-experimental-subgroup-matrix", ...(profiling() ? ["timestamp-query"] : [])] as GPUFeatureName[];
  const optional = wanted.filter((f) => adapter.features.has(f));
  return adapter.requestDevice({
    requiredFeatures: ["shader-f16", ...optional],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
    },
  });
}

/** The manifest and every tensor of `base` (a URL ending in "/"), uploaded to the GPU. */
export async function loadWeights<M extends BaseManifest>(base: string, opts: GpuOptions = {}) {
  const get = opts.fetch ?? globalThis.fetch.bind(globalThis);
  const device = opts.device ?? (await requestDevice());
  const m = (await (await get(`${base}manifest.json`)).json()) as M;
  const w: Record<string, Weight> = {};
  let loaded = 0;
  const upload = async (offset: number, bytes: number) => {
    const res = await get(`${base}weights.bin`, { headers: { Range: `bytes=${offset}-${offset + bytes - 1}` } });
    const data = new Uint8Array(await res.arrayBuffer());
    if (data.byteLength !== bytes) throw new Error(`decision.js: range read returned ${data.byteLength} bytes, wanted ${bytes} (does the server support Range requests?)`);
    const buf = device.createBuffer({ size: align4(bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const whole = bytes - (bytes % 4);
    if (whole) device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, whole);
    if (bytes % 4) device.queue.writeBuffer(buf, whole, new Uint8Array([...data.subarray(whole), 0, 0, 0].slice(0, 4)));
    loaded += bytes;
    opts.onProgress?.(loaded, m.bytes);
    return buf;
  };
  const names = Object.keys(m.tensors);
  const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  const largest = Math.max(...names.map((n) => m.tensors[n]!.bytes));
  if (largest > limit) throw new Error(`decision.js: this GPU's buffers are limited to ${(limit / 1e6).toFixed(0)} MB, this model needs ${(largest / 1e6).toFixed(0)} MB`);
  for (let i = 0; i < names.length; i += 6) {
    await Promise.all(names.slice(i, i + 6).map(async (name) => {
      const t = m.tensors[name]!;
      w[name] = t.dtype === "q4" || t.dtype === "q8"
        ? { codes: await upload(t.offset, t.bytes), scales: await upload(t.scales!, t.scales_bytes!), shape: t.shape, bits: t.dtype === "q8" ? 8 : 4 }
        : { data: await upload(t.offset, t.bytes), shape: t.shape };
    }));
  }
  return { device, m, w };
}

export class GpuEngine {
  private pipelines = new Map<string, GPUComputePipeline>();
  /** Whether the matrix-unit kernels are in use. */
  readonly matrixUnits: boolean;
  /** Whether subgroup operations are available (the tiled attention needs them). */
  readonly subgroups: boolean;
  stats = { forwards: 0, tokens: 0, gpuMs: 0 };
  /** Whether the 4-way split-K matmul fits in the device's workgroup memory (24 KB). */
  private readonly splitK4: boolean;
  private readonly splitKernels = new Map<string, string>();

  constructor(readonly device: GPUDevice, readonly w: Record<string, Weight>, opts: GpuOptions = {}) {
    this.matrixUnits = device.features.has("chromium-experimental-subgroup-matrix") && opts.matrixUnits !== false;
    this.subgroups = device.features.has("subgroups");
    this.splitK4 = device.limits.maxComputeWorkgroupStorageSize >= 24576;
  }

  /** Frees every weight. */
  destroy() {
    for (const wt of Object.values(this.w)) for (const b of [wt.codes, wt.scales, wt.data, wt.tiled, wt.cols4]) b?.destroy();
  }

  /** Converts the matmul weights for the direct kernel (see GpuOptions.directWeights). `keep`: weights also
   *  read in their quantized form (e.g. a gathered embedding). */
  protected async tileWeights(opts: GpuOptions, keep: string[] = []) {
    const mu = this.matrixUnits;
    const names = Object.keys(this.w).filter((n) => {
      const wt = this.w[n]!, [N, K] = wt.shape as [number, number];
      return wt.bits && !keep.includes(n) && wt.shape.length === 2 && N % (mu ? 64 : 128) === 0 && K % 32 === 0;
    });
    const bytes = names.reduce((a, n) => a + this.w[n]!.shape[0]! * this.w[n]!.shape[1]! * 2, 0);
    if (!(opts.directWeights ?? (bytes < 1.5e9 && !mobile()))) return;
    for (const n of names) {
      const wt = this.w[n]!, [N, K] = wt.shape as [number, number];
      const out = this.device.createBuffer({ size: N * K * 2, usage: GPUBufferUsage.STORAGE });
      const swiglu = mu && n.endsWith("w13") && (N / 2) % 32 === 0;
      const { keep: k, dispatch, finish } = this.recorder();
      dispatch(mu ? tileWeights(wt.bits!, swiglu) : tileWeightsPortable(wt.bits!), [k(this.uniform([N, K])), wt.codes!, wt.scales!, out], Math.ceil((mu ? N * K : (N * K) / 4) / 256));
      await finish(0);
      wt.codes!.destroy(); wt.scales!.destroy();
      Object.assign(wt, mu ? { tiled: out, swiglu } : { cols4: out }, { codes: undefined, scales: undefined });
    }
    this.stats.forwards = 0;
  }

  protected pipeline(code: string) {
    let p = this.pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      void module.getCompilationInfo().then((info) => {
        for (const m of info.messages) if (m.type === "error") console.error(`decision.js WGSL ${m.lineNum}:${m.linePos} ${m.message}\n${code.split("\n")[m.lineNum - 1]}`);
      });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  protected storage(data: ArrayBufferView) {
    const buf = this.device.createBuffer({ size: Math.max(16, align4(data.byteLength)), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
    return buf;
  }

  protected uniform(values: number[], floats: number[] = []) {
    const ab = new ArrayBuffer(32);
    const u = new Uint32Array(ab, 0, 4);
    values.forEach((v, i) => (u[i] = v));
    const f = new Float32Array(ab, 16, 4);
    floats.forEach((v, i) => (f[i] = v));
    const buf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, ab);
    return buf;
  }

  /** f16 activation buffer of `rows` × `cols` elements, rounded up to whole 64-row tiles (the direct
   *  matmul reads whole tiles). */
  protected activationRows(rows: number, cols: number) {
    return this.activations(Math.ceil(rows / 64) * 64 * cols);
  }

  /** f16 activation buffer of `elems` elements. */
  protected activations(elems: number) {
    return this.device.createBuffer({ size: align4(elems * 2), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  /** A forward pass recorder: temporary buffers die with it; `finish` submits and waits. */
  protected recorder() {
    const d = this.device;
    const temps: GPUBuffer[] = [];
    const keep = <X extends GPUBuffer>(b: X) => (temps.push(b), b);
    d.pushErrorScope("validation"); // a shader or binding error fails the call instead of returning garbage
    const enc = d.createCommandEncoder();
    let pass = enc.beginComputePass();
    const skip = (globalThis as { __d1Skip?: string[] }).__d1Skip ?? [];
    const prof = d.features.has("timestamp-query") ? profiling() : undefined;
    const timed: string[] = [];
    const qs = prof && d.createQuerySet({ type: "timestamp", count: 4096 });
    const dispatch = (code: string, entries: GPUBuffer[], x: number, y = 1) => {
      if (skip.some((k) => code.includes(k))) return; // profiling: drop one kind of kernel
      if (y === 1 && x > 65535) [x, y] = [65535, Math.ceil(x / 65535)]; // 2-D grid for long flat dispatches
      const pipe = this.pipeline(code);
      const bg = d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      if (qs && timed.length < 2048) {
        pass.end();
        pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * timed.length, endOfPassWriteIndex: 2 * timed.length + 1 } });
        timed.push(code);
      }
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(x, y);
    };
    const matmul = (X: GPUBuffer, wt: Weight, Y: GPUBuffer, M: number) => {
      const [N, K] = wt.shape as [number, number];
      if (wt.tiled) {
        // few rows: split K between subgroups (4 ways up to 128 rows, 2 up to 512; Apple M5 Pro)
        const sk = (globalThis as { __d1SplitK?: { tm: number; tn: number; ks: number; below: number } }).__d1SplitK // tuning
          ?? (M <= 128 && this.splitK4 ? { tm: 32, tn: 32, ks: 4, below: Infinity } : M <= 512 ? { tm: 32, tn: 32, ks: 2, below: Infinity } : undefined);
        if (sk && M < sk.below && K % (8 * sk.ks) === 0 && N % sk.tn === 0) {
          const code = `${sk.tm},${sk.tn},${sk.ks}`;
          const kernel = this.splitKernels.get(code) ?? this.splitKernels.set(code, matmulSplitK(sk)).get(code)!;
          return dispatch(kernel, [keep(this.uniform([M, N, K])), X, wt.tiled, Y], N / sk.tn, Math.ceil(M / sk.tm));
        }
        const t = (globalThis as { __d1Direct?: { tm: number; tn: number; sm: number; sn: number; flush?: number } }).__d1Direct; // tuning
        const code = t ? matmulDirect(t).replace("FLUSH: u32 = 128u", `FLUSH: u32 = ${t.flush ?? 128}u`) : MATMUL_DIRECT;
        return dispatch(code, [keep(this.uniform([M, N, K])), X, wt.tiled, Y], Math.ceil(N / (t ?? DIRECT_TILE).tn), Math.ceil(M / (t ?? DIRECT_TILE).tm));
      }
      if (wt.cols4) {
        const t = (globalThis as { __d1PortableF16?: typeof PORTABLE_F16_TILE }).__d1PortableF16; // tuning
        return dispatch(t ? matmulPortableF16(t) : MATMUL_PORTABLE_F16, [keep(this.uniform([M, N, K])), X, wt.cols4, Y], Math.ceil(N / (t ?? PORTABLE_F16_TILE).tn), Math.ceil(M / (t ?? PORTABLE_F16_TILE).tm));
      }
      let kernel = this.matrixUnits && N % 64 === 0 ? (wt.bits === 8 ? MATMUL_Q8_SG32 : MATMUL_Q4_SG32) : wt.bits === 8 ? MATMUL_PORTABLE_Q8 : MATMUL_PORTABLE_Q4;
      let [tm, tn] = kernel === MATMUL_Q4_SG32 || kernel === MATMUL_Q8_SG32 ? [64, 64] : [64, 128];
      const exp = (globalThis as { __d1Portable?: { tm: number; tn: number; rm: number; rn: number } }).__d1Portable; // tuning
      if (exp && !this.matrixUnits) { kernel = matmulPortable({ ...exp, bits: wt.bits === 8 ? 8 : 4 }); [tm, tn] = [exp.tm, exp.tn]; }
      dispatch(kernel, [keep(this.uniform([M, N, K])), X, wt.codes!, wt.scales!, Y], Math.ceil(N / tn), Math.ceil(M / tm));
    };
    /** H = silu(X · W1ᵀ) · (X · W3ᵀ) for a fused w1 | w3: one kernel on a swiglu weight, else the matmul into Z
     *  then SILU_MUL. */
    const swiglu = (X: GPUBuffer, wt: Weight, Z: GPUBuffer, H: GPUBuffer, M: number) => {
      const [N, K] = wt.shape as [number, number];
      if (wt.swiglu) return dispatch(MATMUL_SWIGLU, [keep(this.uniform([M, N, K])), X, wt.tiled!, H], N / DIRECT_TILE.tn, Math.ceil(M / DIRECT_TILE.tm));
      matmul(X, wt, Z, M);
      dispatch(SILU_MUL, [keep(this.uniform([M, N / 2])), Z, H], Math.ceil((M * N) / 2 / 256));
    };
    /** Copy a buffer out mid-pass and get it back as floats after `finish`. */
    const snapshots: [GPUBuffer, number, "f16" | "f32"][] = [];
    const snapshot = (src: GPUBuffer, bytes: number, type: "f16" | "f32" = "f16") => {
      pass.end();
      const b = keep(d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
      enc.copyBufferToBuffer(src, 0, b, 0, bytes);
      pass = enc.beginComputePass();
      snapshots.push([b, bytes, type]);
      return snapshots.length - 1;
    };
    const finish = async (tokens: number): Promise<Float32Array[]> => {
      pass.end();
      let times: GPUBuffer | undefined;
      if (qs && timed.length) {
        const resolved = keep(d.createBuffer({ size: timed.length * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }));
        times = keep(d.createBuffer({ size: timed.length * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
        enc.resolveQuerySet(qs, 0, timed.length * 2, resolved, 0);
        enc.copyBufferToBuffer(resolved, 0, times, 0, timed.length * 16);
      }
      const t0 = performance.now();
      d.queue.submit([enc.finish()]);
      const error = await d.popErrorScope();
      if (error) {
        temps.forEach((b) => b.destroy());
        qs?.destroy();
        throw new Error(`decision.js: WebGPU error: ${error.message}`);
      }
      const out: Float32Array[] = [];
      for (const [b, , type] of snapshots) {
        await b.mapAsync(GPUMapMode.READ);
        const raw = b.getMappedRange().slice(0);
        out.push(type === "f16" ? f16ToF32(new Uint16Array(raw)) : new Float32Array(raw));
      }
      if (times) {
        await times.mapAsync(GPUMapMode.READ);
        const ns = new BigInt64Array(times.getMappedRange());
        timed.forEach((code, i) => prof!.set(code, (prof!.get(code) ?? 0) + Number(ns[2 * i + 1]! - ns[2 * i]!)));
      }
      this.stats.gpuMs += performance.now() - t0;
      this.stats.forwards++;
      this.stats.tokens += tokens;
      temps.forEach((b) => b.destroy());
      qs?.destroy();
      return out;
    };
    return { keep, dispatch, matmul, swiglu, snapshot, finish };
  }
}

/** IEEE half → float, without relying on Float16Array. */
function f16ToF32(h: Uint16Array) {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = h[i]!, s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 31, f = x & 1023;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

export const softmax = (s: number[]) => {
  const m = Math.max(...s), e = s.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / z);
};
