// A WebGPU inference engine for d3-lite (vllm-sr/d3-lite): a Qwen3.5-0.8B backbone (24 layers: 18 Gated
// DeltaNet, every fourth a gated attention layer with 256-dim heads, bidirectional within the sequence as
// d3's "noncausal_full_attention" mode) and a code readout at the last token. One question = one sequence;
// many sequences are packed on one token axis and run in a single command buffer. Weights: scripts/export-d3.py.
import { ADD_RMSNORM, GATHER_Q4, GATHER_Q8 } from "./kernels.js";
import { attention256, QK_NORM_ROPE_256, READOUT, gdnLayer } from "./d3-kernels.js";
import { GpuEngine, loadWeights, type BaseManifest, type GpuOptions, type Weight } from "./gpu.js";

export interface D3Manifest extends BaseManifest {
  config: {
    model: "d3"; hidden: number; intermediate: number; layer_types: string[]; eps: number;
    heads: number; kv_heads: number; head_dim: number; rotary_dim: number; rope_theta: number;
    linear_heads: number; linear_key_dim: number; linear_value_dim: number; conv_kernel: number; linear_rows: number;
    codes: string[]; temperature: number;
  };
}

/** One question: its prompt's token ids and its number of options (codes 0 … n-1 are read). */
export interface D3Seq { ids: number[]; options: number }

export class D3Engine extends GpuEngine {
  private bufs: Record<string, GPUBuffer> = {};
  private capacity = 0;
  /** The DeltaNet mixer kernel: 4 lanes per state row when the device allows 512-thread workgroups, else 2. */
  private readonly gdn: string;
  /** The attention kernel: queries × query heads of one key/value head per workgroup (see attention256). */
  private readonly attn: string;
  private readonly attnTile: { qt: number; g: number };
  private constructor(device: GPUDevice, readonly m: D3Manifest, w: Record<string, Weight>, opts: GpuOptions) {
    super(device, w, opts);
    if (!this.subgroups) throw new Error("decision.js: d3-lite needs WebGPU subgroups");
    const big = Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX) >= 512;
    const tune = globalThis as { __d3Lanes?: 2 | 4; __d3GdnForm?: "mlx" | "ort" };
    this.gdn = gdnLayer(tune.__d3Lanes ?? (big ? 4 : 2), tune.__d3GdnForm ?? "ort"); // tuning: __d3Lanes, __d3GdnForm
    const patch = (globalThis as { __d3GdnPatch?: (code: string) => string }).__d3GdnPatch; // profiling experiments
    if (patch) this.gdn = patch(this.gdn);
    this.attnTile = (globalThis as { __d3Attn?: { qt: number; g: number } }).__d3Attn ?? { qt: 8, g: 2 }; // tuning
    this.attn = attention256(this.attnTile.qt, this.attnTile.g);
  }

  /** Weights exported by scripts/export-d3.py, at `base` (a URL ending in "/"). */
  static async load(base: string, opts: GpuOptions = {}): Promise<D3Engine> {
    const { device, m, w } = await loadWeights<D3Manifest>(base, opts);
    const engine = new D3Engine(device, m, w, opts);
    await engine.tileWeights(opts, ["embed"]); // the embedding is only gathered
    return engine;
  }

  private ensure(T: number) {
    if (T <= this.capacity) return;
    for (const b of Object.values(this.bufs)) b.destroy();
    const c = this.m.config, H = c.hidden, LH = c.linear_heads;
    const cap = Math.max(T, 256);
    const mk = (n: number) => this.activationRows(cap, n);
    this.bufs = {
      x: mk(H), normed: mk(H), o: mk(H),
      z: mk(Math.max(c.linear_rows, 2 * c.intermediate, (c.heads * 2 + c.kv_heads * 2) * c.head_dim)),
      y: mk(Math.max(LH * c.linear_value_dim, c.heads * c.head_dim)), g: mk(LH * c.linear_value_dim), h: mk(c.intermediate),
    };
    this.capacity = cap;
  }

  /** Code logits per sequence (its first `options` codes), before the temperature. `groups` (sequences per
   *  request, consecutive): share each request's common token prefix (system prompt and state) as one trunk,
   *  read once; its tokens then see only the trunk, and each question sees the trunk and itself. An
   *  approximation of d3 (its state normally sees the question), off by default. */
  async logits(seqs: D3Seq[], debugLayers: number[] = [], groups?: number[]): Promise<{ logits: number[][]; debug: Record<number, Float32Array> }> {
    const c = this.m.config, H = c.hidden, I = c.intermediate, LH = c.linear_heads, eps = c.eps;
    // per token: meta (start, end, -1, position) and the key ranges of the attention; per scan: two spans
    const ids: number[] = [], meta: number[] = [], rng: number[] = [], spans: number[] = [], pairs: number[] = [];
    const sizes = groups ?? seqs.map(() => 1);
    let si = 0;
    for (const n of sizes) {
      const group = seqs.slice(si, si + n);
      si += n;
      let P = 0; // the common prefix, at least one token left per question
      if (n > 1) {
        const lim = Math.min(...group.map((s) => s.ids.length)) - 1;
        while (P < lim && group.every((s) => s.ids[P] === group[0]!.ids[P])) P++;
      }
      const ts = ids.length;
      for (let i = 0; i < P; i++) (ids.push(group[0]!.ids[i]!), meta.push(ts, ts + P, -1, i), rng.push(ts, ts + P, 0, 0));
      group.forEach((s, b) => {
        const bs = ids.length, be = bs + s.ids.length - P;
        for (let i = P; i < s.ids.length; i++) {
          ids.push(s.ids[i]!); meta.push(bs, be, -1, i);
          rng.push(P ? ts : bs, P ? ts + P : be, P ? bs : 0, P ? be : 0);
        }
        spans.push(P ? ts : bs, (P ? P : be - bs) | (b === 0 || !P ? 0x80000000 : 0), P ? bs : 0, P ? be - bs : 0);
        for (let k = 0; k < s.options; k++) pairs.push(be - 1, k);
      });
    }
    const T = ids.length, P = pairs.length / 2;
    this.ensure(T);
    const B = this.bufs;
    const { keep, dispatch, matmul, swiglu, snapshot, finish } = this.recorder();
    const idsBuf = keep(this.storage(new Uint32Array(ids)));
    // debugging: causal attention, to compare with a plain transformers forward (d3's noncausal hook off)
    const causal = (globalThis as { __d3Causal?: boolean }).__d3Causal;
    const metaBuf = keep(this.storage(new Int32Array(meta)));
    const rngBuf = keep(this.storage(new Int32Array(causal ? rng.map((v, j) => (j % 4 === 1 ? meta[j - 1]! + meta[j + 2]! + 1 : v)) : rng)));
    const spansBuf = keep(this.storage(new Uint32Array(spans)));
    const pairsBuf = keep(this.storage(new Uint32Array(pairs)));
    const outBuf = keep(this.device.createBuffer({ size: Math.max(16, P * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
    const w = (name: string) => this.w[name]!;
    const flat = (n: number, size = 256) => Math.ceil(n / size);
    const rms = (add: boolean, weight: string) => dispatch(ADD_RMSNORM, [keep(this.uniform([T, H, add ? 1 : 0], [eps])), B.x!, B.o!, w(weight).data!, B.normed!], T);
    const snaps: [number, number][] = [];

    dispatch(w("embed").bits === 8 ? GATHER_Q8 : GATHER_Q4, [keep(this.uniform([T, H])), idsBuf, w("embed").codes!, w("embed").scales!, B.x!], flat(T * H));
    for (let i = 0; i < c.layer_types.length; i++) {
      const p = `layers.${i}.`;
      rms(i > 0, p + "input_norm"); // adds the previous layer's MLP output
      if (debugLayers.includes(i)) snaps.push([i, snapshot(B.x!, T * H * 2)]); // = hidden_states[i]
      if (c.layer_types[i] === "linear_attention") {
        const rows = c.linear_rows, qkv = 3 * LH * 128;
        matmul(B.normed!, w(p + "in"), B.z!, T);
        dispatch(this.gdn, [keep(this.uniform([seqs.length, LH, rows, qkv], [eps])), spansBuf, B.z!, w(p + "conv").data!, w(p + "A_log").data!, w(p + "dt_bias").data!, w(p + "gnorm").data!, B.g!], 1, (spans.length / 4) * LH);
        matmul(B.g!, w(p + "out"), B.o!, T);
      } else {
        const stride = (2 * c.heads + 2 * c.kv_heads) * c.head_dim;
        matmul(B.normed!, w(p + "qkvg"), B.z!, T);
        dispatch(QK_NORM_ROPE_256, [keep(this.uniform([T, c.heads, c.kv_heads, stride], [eps, c.rope_theta])), B.z!, w(p + "q_norm").data!, w(p + "k_norm").data!, metaBuf], flat(T * (c.heads + c.kv_heads), 64));
        dispatch(this.attn, [keep(this.uniform([T, c.heads, c.kv_heads, stride], [1 / Math.sqrt(c.head_dim)])), B.z!, rngBuf, B.y!], Math.ceil(T / this.attnTile.qt), c.heads / this.attnTile.g);
        matmul(B.y!, w(p + "o"), B.o!, T);
      }
      rms(true, p + "post_norm");
      swiglu(B.normed!, w(p + "w13"), B.z!, B.h!, T);
      matmul(B.h!, w(p + "w2"), B.o!, T);
    }
    rms(true, "final_norm");
    if (debugLayers.includes(-1)) snaps.push([-1, snapshot(B.normed!, T * H * 2)]); // the backbone's output
    dispatch(READOUT, [keep(this.uniform([P, H])), B.normed!, w("readout").data!, pairsBuf, outBuf], P);
    const outIdx = snapshot(outBuf, P * 4, "f32");
    const got = await finish(T);
    const out = got[outIdx]!;
    let k = 0;
    return { logits: seqs.map((s) => Array.from({ length: s.options }, () => out[k++]!)), debug: Object.fromEntries(snaps.map(([l, j]) => [l, got[j]!])) };
  }
}
