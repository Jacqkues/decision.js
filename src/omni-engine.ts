// A WebGPU inference engine for LiquidAI d1-omni-600M text decisions. The trunk is a bidirectional
// LFM2.5 encoder (centred 3-tap convolutions, GQA attention over the whole sequence); the head adds a
// question-type embedding, runs two pre-norm transformer layers and scores each option marker with an MLP.
// One question = one sequence (the encoder reads the state anew for each question); many sequences are
// packed on one token axis and run in a single command buffer. Weights: scripts/export-omni.py (q8 or q4).
import { ADD_LAYERNORM, ADD_RMSNORM, ADD_ROWS, ATTENTION, ATTENTION_TILED, BIAS_ACT, GATHER_Q4, GATHER_Q8, GATHER_ROWS, QK_NORM_ROPE, SCORE, SHORT_CONV } from "./kernels.js";
import { GpuEngine, loadWeights, softmax, type BaseManifest, type GpuOptions, type Weight } from "./gpu.js";

export interface OmniManifest extends BaseManifest {
  config: {
    hidden: number; intermediate: number; heads: number; kv_heads: number; vocab: number; eps: number; rope_theta: number; layer_types: string[];
    head_layers: number; head_heads: number; head_ff: number; ln_eps: number; temperatures: Record<string, number>;
  };
}

export type QType = "choice" | "score" | "noul";
const QTYPES: Record<QType, number> = { choice: 0, score: 1, noul: 2 };

/** One question over one state, encoded as d1-omni's prompt.py does: token ids and one marker per option. */
export interface OmniSeq { ids: number[]; markers: number[]; type: QType }

/** The calibration temperature of a question (prompt.py temperature_key). */
export function temperature(m: OmniManifest["config"], type: QType, options: number) {
  const k = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+";
  return m.temperatures[`${type}:${k}`] ?? m.temperatures[type] ?? 1;
}

export class D1OmniEngine extends GpuEngine {
  private bufs: Record<string, GPUBuffer> = {};
  private capacity = 0;

  private constructor(device: GPUDevice, readonly m: OmniManifest, w: Record<string, Weight>, opts: GpuOptions) {
    super(device, w, opts);
  }

  /** Weights exported by scripts/export-omni.py, at `base` (a URL ending in "/"). */
  static async load(base: string, opts: GpuOptions = {}): Promise<D1OmniEngine> {
    const { device, m, w } = await loadWeights<OmniManifest>(base, opts);
    const engine = new D1OmniEngine(device, m, w, opts);
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

  /** Raw marker logits per sequence (before temperature), and optional debug snapshots. */
  async logits(seqs: OmniSeq[], debug = false): Promise<{ logits: number[][]; debug: Record<string, Float32Array> }> {
    const c = this.m.config;
    const { hidden: H, intermediate: I, heads, kv_heads: KV, eps, rope_theta, layer_types } = c;
    // ---- packing, on the host
    const ids: number[] = [], pos: number[] = [], taps: number[] = [], meta: number[] = [], qtype: number[] = [], markers: number[] = [];
    for (const s of seqs) {
      const start = ids.length, end = start + s.ids.length;
      s.ids.forEach((id, i) => {
        const t = start + i;
        ids.push(id); pos.push(i); qtype.push(QTYPES[s.type]);
        taps.push(t > start ? t - 1 : -1, t, t + 1 < end ? t + 1 : -1, 0); // centred, never across sequences
        meta.push(start, end, -1, i);
      });
      s.markers.forEach((m) => markers.push(start + m));
    }
    const T = ids.length, M = markers.length;
    this.ensure(T);
    const B = this.bufs;
    const { keep, dispatch, matmul, swiglu, snapshot, finish } = this.recorder();
    const idsBuf = keep(this.storage(new Uint32Array(ids)));
    const posBuf = keep(this.storage(new Uint32Array(pos)));
    const tapsBuf = keep(this.storage(new Int32Array(taps)));
    const metaBuf = keep(this.storage(new Int32Array(meta)));
    const qtypeBuf = keep(this.storage(new Uint32Array(qtype)));
    const markersBuf = keep(this.storage(new Uint32Array(markers)));
    const g = keep(this.activationRows(M, H)), gf = keep(this.activationRows(M, H));
    const outBuf = keep(this.device.createBuffer({ size: Math.max(16, M * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
    const w = (name: string) => this.w[name]!;
    const flat = (n: number) => Math.ceil(n / 256);
    const rms = (add: boolean, weight: string) => dispatch(ADD_RMSNORM, [keep(this.uniform([T, H, add ? 1 : 0], [eps])), B.x!, B.o!, w(weight).data!, B.normed!], T);
    const ln = (X: GPUBuffer, rows: number, add: boolean, name: string, O: GPUBuffer) =>
      dispatch(ADD_LAYERNORM, [keep(this.uniform([rows, H, add ? 1 : 0], [c.ln_eps])), X, B.o!, w(name).data!, w(name + "_bias").data!, O], rows);
    const bias = (Y: GPUBuffer, rows: number, name: string, act: 0 | 1 | 2 = 0) => {
      const b = w(name);
      dispatch(BIAS_ACT, [keep(this.uniform([rows, b.shape[0]!, act])), Y, b.data!], flat(rows * b.shape[0]!));
    };
    const snaps: Record<string, number> = {};
    // tiled attention needs subgroups; else one thread per (token, head)
    const tiled = this.subgroups && (globalThis as { __d1Attn?: string }).__d1Attn !== "naive";
    const attention = (nh: number, nkv: number) => tiled
      ? dispatch(ATTENTION_TILED, [keep(this.uniform([T, nh, nkv, 1], [1 / 8])), B.z!, metaBuf, B.y!], Math.ceil(T / 32), nh)
      : dispatch(ATTENTION, [keep(this.uniform([T, nh, nkv, 1], [1 / 8])), B.z!, metaBuf, B.y!], Math.ceil((T * nh) / 64));

    // ---- trunk
    dispatch(w("embed").bits === 8 ? GATHER_Q8 : GATHER_Q4, [keep(this.uniform([T, H])), idsBuf, w("embed").codes!, w("embed").scales!, B.x!], flat(T * H));
    rms(false, "layers.0.operator_norm");
    for (let i = 0; i < layer_types.length; i++) {
      const p = `layers.${i}.`;
      if (layer_types[i] === "conv") {
        matmul(B.normed!, w(p + "in_proj"), B.z!, T);
        dispatch(SHORT_CONV, [keep(this.uniform([T, H])), B.z!, w(p + "conv").data!, tapsBuf, B.y!], flat(T * H));
        matmul(B.y!, w(p + "out_proj"), B.o!, T);
      } else {
        matmul(B.normed!, w(p + "qkv"), B.z!, T);
        dispatch(QK_NORM_ROPE, [keep(this.uniform([T, heads, KV], [eps, rope_theta])), B.z!, w(p + "q_norm").data!, w(p + "k_norm").data!, posBuf], Math.ceil((T * (heads + KV)) / 64));
        attention(heads, KV);
        matmul(B.y!, w(p + "o_proj"), B.o!, T);
      }
      rms(true, p + "ffn_norm");
      swiglu(B.normed!, w(p + "w13"), B.z!, B.h!, T);
      matmul(B.h!, w(p + "w2"), B.o!, T);
      rms(true, i + 1 < layer_types.length ? `layers.${i + 1}.operator_norm` : "final_norm");
    }
    if (debug) snaps.trunk = snapshot(B.normed!, T * H * 2);

    // ---- head: x = trunk output + type embedding, then pre-norm layers (attention over the whole sequence)
    dispatch(ADD_ROWS, [keep(this.uniform([T, H])), B.normed!, qtypeBuf, w("head.type_emb").data!, B.x!], flat(T * H));
    const hh = c.head_heads;
    for (let i = 0; i < c.head_layers; i++) {
      const p = `head.${i}.`;
      ln(B.x!, T, i > 0, p + "norm1", B.normed!); // i > 0: first adds the previous layer's FFN output
      matmul(B.normed!, w(p + "in_proj"), B.z!, T);
      bias(B.z!, T, p + "in_proj_bias");
      attention(hh, hh);
      matmul(B.y!, w(p + "out_proj"), B.o!, T);
      bias(B.o!, T, p + "out_proj_bias");
      ln(B.x!, T, true, p + "norm2", B.normed!);
      matmul(B.normed!, w(p + "linear1"), B.h!, T);
      bias(B.h!, T, p + "linear1_bias", 1);
      matmul(B.h!, w(p + "linear2"), B.o!, T);
      bias(B.o!, T, p + "linear2_bias");
    }
    // the last residual add, with the scorer's LayerNorm
    ln(B.x!, T, true, "scorer.norm", B.normed!);
    if (debug) snaps.head = snapshot(B.x!, T * H * 2);

    // ---- scorer: LayerNorm → Linear → GELU → Linear(1) at each marker
    dispatch(GATHER_ROWS, [keep(this.uniform([M, H])), B.normed!, markersBuf, g], flat(M * H));
    matmul(g, w("scorer.fc"), gf, M);
    bias(gf, M, "scorer.fc_bias", 2);
    dispatch(SCORE, [keep(this.uniform([M, H])), gf, w("scorer.out").data!, w("scorer.out_bias").data!, outBuf], M);
    const outIdx = snapshot(outBuf, M * 4, "f32");
    const got = await finish(T);
    const out = got[outIdx]!;
    let k = 0;
    const logits = seqs.map((s) => s.markers.map(() => out[k++]!));
    return { logits, debug: Object.fromEntries(Object.entries(snaps).map(([name, i]) => [name, got[i]!])) };
  }

  /** Calibrated probabilities per sequence, in option order (a noul is [false, true], as the graph reads it). */
  async probabilities(seqs: OmniSeq[]): Promise<number[][]> {
    const { logits } = await this.logits(seqs);
    return logits.map((z, i) => {
      const t = temperature(this.m.config, seqs[i]!.type, z.length);
      return softmax(z.map((v) => v / t));
    });
  }
}
