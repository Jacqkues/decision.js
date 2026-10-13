// A WebGPU inference engine for LiquidAI d1-3B decisions (LFM2 backbone), built for one job: many short
// decisions over many rows. Rows are packed as a forest: each row's state is a trunk read once, each of
// its questions a branch continuing it (as d1's own tree runtime); only the option tokens' logits are
// computed at each branch's last token. Weights: scripts/export-3b.py (q4 linears, f16 norms).
import { ADD_RMSNORM, ATTENTION, GATHER_Q4, LEAF_LOGITS, QK_NORM_ROPE, SHORT_CONV } from "./kernels.js";
import { GpuEngine, loadWeights, type BaseManifest, type GpuOptions, type Weight } from "./gpu.js";

export interface D13BManifest extends BaseManifest {
  config: { hidden: number; intermediate: number; heads: number; kv_heads: number; head_dim: number; vocab: number; eps: number; rope_theta: number; layer_types: string[] };
}

/** One row: its state tokens (trunk), one token list per question (branches), and per question the
 *  option token groups to score (each option = max over its group, as d1's readout). */
export interface Tree { trunk: number[]; branches: number[][]; groups: number[][][] }

export class D13BEngine extends GpuEngine {
  private bufs: Record<string, GPUBuffer> = {};
  private capacity = 0;

  private constructor(device: GPUDevice, readonly m: D13BManifest, w: Record<string, Weight>, opts: GpuOptions) {
    super(device, w, opts);
  }

  /** Weights exported by scripts/export-3b.py, at `base` (a URL ending in "/"). */
  static async load(base: string, opts: GpuOptions = {}): Promise<D13BEngine> {
    const { device, m, w } = await loadWeights<D13BManifest>(base, opts);
    if (w.embed?.bits !== 4) throw new Error("decision.js: d1-3B weights must be q4");
    const engine = new D13BEngine(device, m, w, opts);
    await engine.tileWeights(opts, ["embed"]); // the embedding is gathered and read as the LM head
    return engine;
  }

  private ensure(T: number) {
    if (T <= this.capacity) return;
    for (const b of Object.values(this.bufs)) b.destroy();
    const { hidden: H, intermediate: I } = this.m.config;
    const cap = Math.max(T, 256);
    const mk = (n: number) => this.activationRows(cap, n);
    this.bufs = { x: mk(H), normed: mk(H), z: mk(2 * I), y: mk(H), h: mk(I), o: mk(H) };
    this.capacity = cap;
  }

  /** Option scores for every question of every tree: [tree][question][option] (max over the option's group). */
  async run(trees: Tree[], debugLayers: number[] = []): Promise<{ scores: number[][][]; debug: Record<number, Float32Array> }> {
    const { hidden: H, intermediate: I, heads, kv_heads: KV, eps, rope_theta, layer_types } = this.m.config;
    // ---- the forest, on the host
    const ids: number[] = [], pos: number[] = [], taps: number[] = [], meta: number[] = [], leaves: number[] = [], leafIds: number[][] = [];
    for (const tr of trees) {
      const start = ids.length;
      const total = tr.trunk.length + tr.branches.reduce((a, b) => a + b.length, 0);
      const p = tr.trunk.length;
      tr.trunk.forEach((id, i) => {
        ids.push(id); pos.push(i);
        taps.push(i >= 2 ? start + i - 2 : -1, i >= 1 ? start + i - 1 : -1, start + i, 0);
        meta.push(start, start + total, -1, i);
      });
      tr.branches.forEach((br, b) => {
        br.forEach((id, o) => {
          const t = ids.length;
          ids.push(id); pos.push(p + o);
          const back = (j: number) => (o - j >= 0 ? t - j : p + (o - j) >= 0 ? start + p + (o - j) : -1);
          taps.push(back(2), back(1), t, 0);
          meta.push(start, start + total, b, p + o);
        });
        leaves.push(ids.length - 1);
        leafIds.push(tr.groups[b]!.flat());
      });
    }
    const T = ids.length, L = leaves.length;
    const nIds = Math.max(...leafIds.map((x) => x.length));
    const flatIds = new Uint32Array(L * nIds);
    leafIds.forEach((g, i) => g.forEach((id, j) => (flatIds[i * nIds + j] = id)));
    this.ensure(T);
    const B = this.bufs;
    const { keep, dispatch, matmul, swiglu, snapshot, finish } = this.recorder();
    const idsBuf = keep(this.storage(new Uint32Array(ids)));
    const posBuf = keep(this.storage(new Uint32Array(pos)));
    const tapsBuf = keep(this.storage(new Int32Array(taps)));
    const metaBuf = keep(this.storage(new Int32Array(meta)));
    const leavesBuf = keep(this.storage(new Uint32Array(leaves)));
    const leafIdsBuf = keep(this.storage(flatIds));
    const outBuf = keep(this.device.createBuffer({ size: L * nIds * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
    const norm = (add: boolean, weight: Weight) => dispatch(ADD_RMSNORM, [keep(this.uniform([T, H, add ? 1 : 0], [eps])), B.x!, B.o!, weight.data!, B.normed!], T);
    const debug: [number, number][] = [];

    dispatch(GATHER_Q4, [keep(this.uniform([T, H])), idsBuf, this.w.embed!.codes!, this.w.embed!.scales!, B.x!], Math.ceil((T * H) / 256));
    norm(false, this.w["layers.0.operator_norm"]!);
    for (let i = 0; i < layer_types.length; i++) {
      const p = `layers.${i}.`;
      if (layer_types[i] === "conv") {
        matmul(B.normed!, this.w[p + "in_proj"]!, B.z!, T);
        dispatch(SHORT_CONV, [keep(this.uniform([T, H])), B.z!, this.w[p + "conv"]!.data!, tapsBuf, B.y!], Math.ceil((T * H) / 256));
        matmul(B.y!, this.w[p + "out_proj"]!, B.o!, T);
      } else {
        matmul(B.normed!, this.w[p + "qkv"]!, B.z!, T);
        dispatch(QK_NORM_ROPE, [keep(this.uniform([T, heads, KV], [eps, rope_theta])), B.z!, this.w[p + "q_norm"]!.data!, this.w[p + "k_norm"]!.data!, posBuf], Math.ceil((T * (heads + KV)) / 64));
        dispatch(ATTENTION, [keep(this.uniform([T, heads, KV, 0], [1 / Math.sqrt(64)])), B.z!, metaBuf, B.y!], Math.ceil((T * heads) / 64));
        matmul(B.y!, this.w[p + "o_proj"]!, B.o!, T);
      }
      norm(true, this.w[p + "ffn_norm"]!);
      swiglu(B.normed!, this.w[p + "w13"]!, B.z!, B.h!, T);
      matmul(B.h!, this.w[p + "w2"]!, B.o!, T);
      norm(true, this.w[i + 1 < layer_types.length ? `layers.${i + 1}.operator_norm` : "final_norm"]!);
      if (debugLayers.includes(i)) debug.push([i, snapshot(B.x!, T * H * 2)]);
    }
    dispatch(LEAF_LOGITS, [keep(this.uniform([L, H, nIds], [eps])), B.x!, leavesBuf, this.w.final_norm!.data!, leafIdsBuf, this.w.embed!.codes!, this.w.embed!.scales!, outBuf], L);
    const outIdx = snapshot(outBuf, L * nIds * 4, "f32");
    const got = await finish(T);
    const out = got[outIdx]!;
    const dbg: Record<number, Float32Array> = Object.fromEntries(debug.map(([layer, k]) => [layer, got[k]!]));
    // ---- readout: each option scores its best form
    const scores: number[][][] = [];
    let leaf = 0;
    for (const tr of trees) {
      scores.push(tr.groups.map((groups) => {
        let j = 0;
        const s = groups.map((g) => Math.max(...g.map(() => out[leaf * nIds + j++]!)));
        leaf++;
        return s;
      }));
    }
    return { scores, debug: dbg };
  }
}

