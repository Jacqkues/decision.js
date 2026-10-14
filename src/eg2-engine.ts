// A WebGPU inference engine for EmbeddingGemma 2 fine-tuned for typed decisions
// (mrm8488/embeddinggemma-2-ft-typed-decision): a 24-layer bidirectional Gemma text encoder (sandwich RMSNorms,
// q/k/v norms, 256-dim heads with a ±512-token window, every sixth layer global with 512-dim heads, GELU MLP,
// per-layer embeddings gating every layer), mean-pooled and projected to 768 dims, then a small head on the host
// (eg2-head.ts) that scores each option's embedding against the state's and the question's. Every text is one
// sequence; many are packed on one token axis and run in a single command buffer. Weights: scripts/export-eg2.py.
import { GATHER_Q4, GATHER_Q8 } from "./kernels.js";
import { attention, GELU_MUL, PLE_MIX, POOL_PROJECT, QKV_NORM_ROPE, SANDWICH } from "./eg2-kernels.js";
import { GpuEngine, loadWeights, type BaseManifest, type GpuOptions, type TensorInfo, type Weight } from "./gpu.js";

type LayerType = "sliding_attention" | "full_attention";
export interface Eg2Manifest extends BaseManifest {
  config: {
    model: "eg2"; hidden: number; intermediate: number; layers: number; heads: number; layer_types: LayerType[]; window: number; eps: number;
    head_dim: Record<LayerType, 256 | 512>; kv_heads: Record<LayerType, number>; rope_theta: Record<LayerType, number>; ple: number;
    layer_scalars: number[]; embedding_dim: number; prompt: string; max_len: number; embed_dim: number;
    question_types: string[]; temperatures: Record<string, number>;
    head: { width: number; ln_eps: number; prior_scale: number; joint_scale: number; gate: number; residual_bias: number };
  };
  /** The head's tensors (f32), read on the host. */
  host: Record<string, TensorInfo>;
}

export class Eg2Engine extends GpuEngine {
  private bufs: Record<string, GPUBuffer> = {};
  private capacity = 0;
  /** The attention kernel per layer type, and its tile (queries, heads per key/value head). */
  private readonly attn: Record<LayerType, { code: string; qt: number; g: number }>;

  private constructor(device: GPUDevice, readonly m: Eg2Manifest, w: Record<string, Weight>, readonly host: Record<string, Float32Array>, opts: GpuOptions) {
    super(device, w, opts);
    if (!this.subgroups) throw new Error("decision.js: EmbeddingGemma 2 needs WebGPU subgroups");
    const c = m.config;
    const tune = (globalThis as { __eg2Attn?: Partial<Record<LayerType, { qt: number; g: number }>> }).__eg2Attn; // tuning
    const tile = (t: LayerType) => {
      const g = tune?.[t]?.g ?? c.heads / c.kv_heads[t], qt = tune?.[t]?.qt ?? 256 / (16 * g);
      return { code: attention(c.head_dim[t], qt, g), qt, g };
    };
    this.attn = { sliding_attention: tile("sliding_attention"), full_attention: tile("full_attention") };
  }

  /** Weights exported by scripts/export-eg2.py, at `base` (a URL ending in "/"). */
  static async load(base: string, opts: GpuOptions = {}): Promise<Eg2Engine> {
    const { device, m, w } = await loadWeights<Eg2Manifest>(base, opts);
    const get = opts.fetch ?? globalThis.fetch.bind(globalThis);
    const host: Record<string, Float32Array> = {};
    for (const [name, t] of Object.entries(m.host)) {
      const res = await get(`${base}weights.bin`, { headers: { Range: `bytes=${t.offset}-${t.offset + t.bytes - 1}` } });
      host[name] = new Float32Array(await res.arrayBuffer());
    }
    const engine = new Eg2Engine(device, m, w, host, opts);
    await engine.tileWeights(opts, ["embed"]); // the embedding is only gathered
    return engine;
  }

  private ensure(T: number) {
    if (T <= this.capacity) return;
    for (const b of Object.values(this.bufs)) b.destroy();
    const c = this.m.config, H = c.hidden;
    const cap = Math.max(T, 256);
    const mk = (n: number) => this.activationRows(cap, n);
    const qkv = Math.max(...c.layer_types.map((t) => (c.heads + 2 * c.kv_heads[t]) * c.head_dim[t]));
    const attn = Math.max(...c.layer_types.map((t) => c.heads * c.head_dim[t]));
    this.bufs = {
      x: mk(H), xe: mk(H), normed: mk(H), o: mk(H), e: mk(c.ple), gate: mk(c.ple), mix: mk(c.ple),
      z: mk(Math.max(qkv, 2 * c.intermediate)), y: mk(attn), h: mk(c.intermediate),
    };
    this.capacity = cap;
  }

  /** The normalized embedding of every token sequence (each one text, with its <bos> … <eos>), f32. */
  async embed(seqs: number[][], debugLayers: number[] = []): Promise<{ embeddings: Float32Array[]; debug: Record<number, Float32Array> }> {
    const c = this.m.config, H = c.hidden, eps = c.eps, N = c.embedding_dim;
    // per token: (start, end, -1, position), and its key range on the global and the sliding layers
    const ids: number[] = [], meta: number[] = [], full: number[] = [], slide: number[] = [], ranges: number[] = [];
    for (const s of seqs) {
      const start = ids.length, end = start + s.length;
      s.forEach((id, i) => {
        const t = start + i;
        ids.push(id); meta.push(start, end, -1, i);
        full.push(start, end, 0, 0);
        slide.push(Math.max(start, t - c.window), Math.min(end, t + c.window + 1), 0, 0); // |query - key| ≤ window
      });
      ranges.push(start, end);
    }
    const T = ids.length, S = seqs.length;
    this.ensure(T);
    const B = this.bufs;
    const { keep, dispatch, matmul, snapshot, finish } = this.recorder();
    const idsBuf = keep(this.storage(new Uint32Array(ids)));
    const metaBuf = keep(this.storage(new Int32Array(meta)));
    const rng = { full_attention: keep(this.storage(new Int32Array(full))), sliding_attention: keep(this.storage(new Int32Array(slide))) };
    const seqBuf = keep(this.storage(new Uint32Array(ranges)));
    const outBuf = keep(this.device.createBuffer({ size: S * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
    const w = (name: string) => this.w[name]!;
    const flat = (n: number, size = 256) => Math.ceil(n / size);
    const zero = keep(this.storage(new Uint16Array(H)));
    /** X = (X + rmsnorm(Y) · w1) · scale, then normed = rmsnorm(X) · w2 when w2 is given. */
    const sandwich = (Y: GPUBuffer, w1: string, w2?: string, scale = 1) =>
      dispatch(SANDWICH, [keep(this.uniform([T, H, w2 ? 1 : 0], [eps, scale])), B.x!, Y, w(w1).data!, w2 ? w(w2).data! : zero, B.normed!], T);
    const snaps: [number, number][] = [];

    // ---- embeddings (√hidden folded in): the residual stream, and a copy each layer projects its own embedding from
    const gather = w("embed").bits === 8 ? GATHER_Q8 : GATHER_Q4;
    for (const X of [B.x!, B.xe!]) dispatch(gather, [keep(this.uniform([T, H])), idsBuf, w("embed").codes!, w("embed").scales!, X], flat(T * H));
    // normed = rmsnorm(x) · input_layernorm of layer 0: the sandwich of a zero branch
    dispatch(SANDWICH, [keep(this.uniform([T, H, 1], [eps, 1])), B.x!, zero, zero, w("layers.0.input_layernorm").data!, B.normed!], T);
    const layers = (globalThis as { __eg2Layers?: number }).__eg2Layers ?? c.layers; // profiling: a truncated encoder
    for (let i = 0; i < layers; i++) {
      const p = `layers.${i}.`, type = c.layer_types[i]!, D = c.head_dim[type], kv = c.kv_heads[type], at = this.attn[type];
      if (debugLayers.includes(i)) snaps.push([i, snapshot(B.x!, T * H * 2)]); // = the encoder's hidden_states[i]
      // attention: x += post_norm(o(attn(rope(norm(q)), rope(norm(k)), norm(v))))
      matmul(B.normed!, w(p + "qkv"), B.z!, T);
      dispatch(QKV_NORM_ROPE, [keep(this.uniform([T, c.heads, kv, D], [eps, c.rope_theta[type]])), B.z!, w(p + "q_norm").data!, w(p + "k_norm").data!, metaBuf], T * (c.heads + 2 * kv));
      dispatch(at.code, [keep(this.uniform([T, c.heads, kv], [1])), B.z!, rng[type], B.y!], Math.ceil(T / at.qt), c.heads / at.g);
      matmul(B.y!, w(p + "o"), B.o!, T);
      sandwich(B.o!, p + "post_attention_layernorm", p + "pre_feedforward_layernorm");
      // MLP: x += post_norm(down(gelu(gate) · up))
      matmul(B.normed!, w(p + "gu"), B.z!, T);
      dispatch(GELU_MUL, [keep(this.uniform([T, c.intermediate])), B.z!, B.h!], flat(T * c.intermediate));
      matmul(B.h!, w(p + "down"), B.o!, T);
      sandwich(B.o!, p + "post_feedforward_layernorm");
      // per-layer embedding: x = (x + norm(proj(gelu(gate(x)) · norm(embedding slice)))) · layer scalar
      matmul(B.x!, w(p + "ple_gate"), B.gate!, T);
      matmul(B.xe!, w(p + "ple_in"), B.e!, T);
      dispatch(PLE_MIX, [keep(this.uniform([T, c.ple], [eps])), B.gate!, B.e!, w("ple.norm").data!, B.mix!], T);
      matmul(B.mix!, w(p + "ple_proj"), B.o!, T);
      sandwich(B.o!, p + "ple_norm", i + 1 < c.layers ? `layers.${i + 1}.input_layernorm` : "final_norm", c.layer_scalars[i]);
    }
    if (debugLayers.includes(-1)) snaps.push([-1, snapshot(B.normed!, T * H * 2)]); // the final norm's output
    dispatch(POOL_PROJECT, [keep(this.uniform([S, H, N])), B.normed!, w("projection").data!, seqBuf, outBuf], S);
    const outIdx = snapshot(outBuf, S * N * 4, "f32");
    const got = await finish(T);
    const out = got[outIdx]!;
    const embeddings = seqs.map((_, s) => {
      const e = out.slice(s * N, s * N + c.embed_dim);
      const n = Math.max(Math.hypot(...e), 1e-12);
      return e.map((v) => v / n);
    });
    return { embeddings, debug: Object.fromEntries(snaps.map(([l, j]) => [l, got[j]!])) };
  }
}
