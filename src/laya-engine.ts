// A WebGPU inference engine for Laya (convaiinnovations/laya): a ModernBERT encoder (bias-free LayerNorms,
// RoPE attention, every third layer global and the others a ±64-token window, GeGLU MLP) and d1-omni's
// decision head (question-type embedding, two pre-norm transformer layers, a scorer at each option marker).
// One question = one sequence; many sequences are packed on one token axis and run in a single command
// buffer. Weights: scripts/export-laya.py (q8 or q4).
import { ADD_LAYERNORM, ADD_ROWS, ATTENTION, ATTENTION_TILED, BIAS_ACT, GATHER_Q4, GATHER_Q8, GATHER_ROWS, SCORE } from "./kernels.js";
import { GELU_MUL, ROPE } from "./encoder-kernels.js";
import { GpuEngine, loadWeights, type BaseManifest, type GpuOptions, type Weight } from "./gpu.js";
import type { OmniSeq, QType } from "./omni-engine.js";

export interface LayaManifest extends BaseManifest {
  config: {
    model: "laya"; hidden: number; intermediate: number; heads: number; layers: number; layer_types: string[]; window: number;
    rope_theta: Record<string, number>; eps: number; head_layers: number; head_heads: number; head_ff: number; ln_eps: number;
    max_len: number; head_max_len: number; temperatures: Record<string, number>;
  };
}

const QTYPES: Record<QType, number> = { choice: 0, score: 1, noul: 2 };

export class LayaEngine extends GpuEngine {
  private bufs: Record<string, GPUBuffer> = {};
  private capacity = 0;
  private zeros!: GPUBuffer; // the bias of the bias-free LayerNorms

  private constructor(device: GPUDevice, readonly m: LayaManifest, w: Record<string, Weight>, opts: GpuOptions) {
    super(device, w, opts);
    this.zeros = this.storage(new Uint16Array(m.config.hidden));
  }

  /** Weights exported by scripts/export-laya.py, at `base` (a URL ending in "/"). */
  static async load(base: string, opts: GpuOptions = {}): Promise<LayaEngine> {
    const { device, m, w } = await loadWeights<LayaManifest>(base, opts);
    const engine = new LayaEngine(device, m, w, opts);
    await engine.tileWeights(opts, ["embed"]); // the embedding is only gathered
    return engine;
  }

  private ensure(T: number) {
    if (T <= this.capacity) return;
    for (const b of Object.values(this.bufs)) b.destroy();
    const { hidden: H, intermediate: I, head_ff } = this.m.config;
    const cap = Math.max(T, 256);
    const mk = (n: number) => this.activationRows(cap, n);
    this.bufs = { x: mk(H), normed: mk(H), z: mk(Math.max(2 * I, 3 * H)), y: mk(H), h: mk(Math.max(I, head_ff)), o: mk(H) };
    this.capacity = cap;
  }

  /** Raw marker logits per sequence (before temperature). */
  async logits(seqs: OmniSeq[], debugLayers: number[] = []): Promise<{ logits: number[][]; debug: Record<number, Float32Array> }> {
    const c = this.m.config;
    const { hidden: H, intermediate: I, heads, eps, window } = c;
    // ---- packing, on the host: per token its sequence's range (global layers) and its window (local layers)
    const ids: number[] = [], pos: number[] = [], meta: number[] = [], win: number[] = [], qtype: number[] = [], markers: number[] = [];
    for (const s of seqs) {
      const start = ids.length, end = start + s.ids.length;
      s.ids.forEach((id, i) => {
        const t = start + i;
        ids.push(id); pos.push(i); qtype.push(QTYPES[s.type]);
        meta.push(start, end, -1, i);
        win.push(Math.max(start, t - window), Math.min(end, t + window + 1), -1, i);
      });
      s.markers.forEach((m) => markers.push(start + m));
    }
    const T = ids.length, M = markers.length;
    this.ensure(T);
    const B = this.bufs;
    const { keep, dispatch, matmul, snapshot, finish } = this.recorder();
    const idsBuf = keep(this.storage(new Uint32Array(ids)));
    const posBuf = keep(this.storage(new Uint32Array(pos)));
    const metaBuf = keep(this.storage(new Int32Array(meta)));
    const winBuf = keep(this.storage(new Int32Array(win)));
    const qtypeBuf = keep(this.storage(new Uint32Array(qtype)));
    const markersBuf = keep(this.storage(new Uint32Array(markers)));
    const g = keep(this.activationRows(M, H)), gf = keep(this.activationRows(M, H));
    const outBuf = keep(this.device.createBuffer({ size: Math.max(16, M * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
    const w = (name: string) => this.w[name]!;
    const flat = (n: number) => Math.ceil(n / 256);
    /** X += o (when add), then O = LayerNorm(X) (bias-free when there is no `<name>_bias`). */
    const ln = (X: GPUBuffer, rows: number, add: boolean, name: string, O: GPUBuffer, lnEps = c.ln_eps) =>
      dispatch(ADD_LAYERNORM, [keep(this.uniform([rows, H, add ? 1 : 0], [lnEps])), X, B.o!, w(name).data!, this.w[name + "_bias"]?.data ?? this.zeros, O], rows);
    const bias = (Y: GPUBuffer, rows: number, name: string, act: 0 | 1 | 2 = 0) => {
      const b = w(name);
      dispatch(BIAS_ACT, [keep(this.uniform([rows, b.shape[0]!, act])), Y, b.data!], flat(rows * b.shape[0]!));
    };
    const snaps: [number, number][] = [];
    const tiled = this.subgroups && (globalThis as { __d1Attn?: string }).__d1Attn !== "naive";
    const attention = (nh: number, ranges: GPUBuffer) => tiled
      ? dispatch(ATTENTION_TILED, [keep(this.uniform([T, nh, nh, 1], [1 / 8])), B.z!, ranges, B.y!], Math.ceil(T / 32), nh)
      : dispatch(ATTENTION, [keep(this.uniform([T, nh, nh, 1], [1 / 8])), B.z!, ranges, B.y!], Math.ceil((T * nh) / 64));

    // ---- encoder: embeddings → LayerNorm, then per layer x += attn(LN(x)); x += mlp(LN(x)) (layer 0: no attn LN)
    dispatch(w("embed").bits === 8 ? GATHER_Q8 : GATHER_Q4, [keep(this.uniform([T, H])), idsBuf, w("embed").codes!, w("embed").scales!, B.normed!], flat(T * H));
    ln(B.normed!, T, false, "embed_norm", B.x!, eps);
    for (let i = 0; i < c.layers; i++) {
      const p = `layers.${i}.`, global = c.layer_types[i] === "full_attention";
      let attnIn = B.x!;
      if (this.w[p + "attn_norm"]) (ln(B.x!, T, true, p + "attn_norm", B.normed!, eps), (attnIn = B.normed!)); // adds the previous MLP
      if (debugLayers.includes(i)) snaps.push([i, snapshot(B.x!, T * H * 2)]); // = the encoder's hidden_states[i]
      matmul(attnIn, w(p + "qkv"), B.z!, T);
      dispatch(ROPE, [keep(this.uniform([T, heads, heads], [c.rope_theta[global ? "full_attention" : "sliding_attention"]!])), B.z!, posBuf], Math.ceil((T * 2 * heads * 32) / 64));
      attention(heads, global ? metaBuf : winBuf);
      matmul(B.y!, w(p + "o"), B.o!, T);
      ln(B.x!, T, true, p + "mlp_norm", B.normed!, eps);
      matmul(B.normed!, w(p + "wi"), B.z!, T);
      dispatch(GELU_MUL, [keep(this.uniform([T, I])), B.z!, B.h!], flat(T * I));
      matmul(B.h!, w(p + "wo"), B.o!, T);
    }
    ln(B.x!, T, true, "final_norm", B.normed!, eps);
    if (debugLayers.includes(-1)) snaps.push([-1, snapshot(B.normed!, T * H * 2)]); // the encoder's output

    // ---- head: x = encoder output + type embedding, then pre-norm layers (attention over the whole sequence)
    dispatch(ADD_ROWS, [keep(this.uniform([T, H])), B.normed!, qtypeBuf, w("head.type_emb").data!, B.x!], flat(T * H));
    for (let i = 0; i < c.head_layers; i++) {
      const p = `head.${i}.`;
      ln(B.x!, T, i > 0, p + "norm1", B.normed!);
      matmul(B.normed!, w(p + "in_proj"), B.z!, T);
      bias(B.z!, T, p + "in_proj_bias");
      attention(c.head_heads, metaBuf);
      matmul(B.y!, w(p + "out_proj"), B.o!, T);
      bias(B.o!, T, p + "out_proj_bias");
      ln(B.x!, T, true, p + "norm2", B.normed!);
      matmul(B.normed!, w(p + "linear1"), B.h!, T);
      bias(B.h!, T, p + "linear1_bias", 1);
      matmul(B.h!, w(p + "linear2"), B.o!, T);
      bias(B.o!, T, p + "linear2_bias");
    }
    ln(B.x!, T, true, "scorer.norm", B.normed!);
    if (debugLayers.includes(-2)) snaps.push([-2, snapshot(B.x!, T * H * 2)]); // the head's output

    // ---- scorer: LayerNorm → Linear → GELU → Linear(1) at each marker
    dispatch(GATHER_ROWS, [keep(this.uniform([M, H])), B.normed!, markersBuf, g], flat(M * H));
    matmul(g, w("scorer.fc"), gf, M);
    bias(gf, M, "scorer.fc_bias", 2);
    dispatch(SCORE, [keep(this.uniform([M, H])), gf, w("scorer.out").data!, w("scorer.out_bias").data!, outBuf], M);
    const outIdx = snapshot(outBuf, M * 4, "f32");
    const got = await finish(T);
    const out = got[outIdx]!;
    let k = 0;
    return { logits: seqs.map((s) => s.markers.map(() => out[k++]!)), debug: Object.fromEntries(snaps.map(([l, i]) => [l, got[i]!])) };
  }
}
