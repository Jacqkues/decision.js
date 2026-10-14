// The decision head of the EmbeddingGemma 2 typed-decision model (GemmaDecisionHead), on the host in f32:
//   f = LayerNorm(field([S, Q, S·Q]) + type)     S, Q, O: the state, question and option embeddings, projected
//   logit_i = prior · cos(o_i, s + q) + gate · (joint · cos(f, LayerNorm(O_i)) + residual([f, Oₙ, f·Oₙ, |f − Oₙ|]))
// The projections of a text (state, question or option) depend on that text only, so they are cached with
// it: a question asked of many states costs only its field and its options' residuals.
import type { Eg2Manifest } from "./eg2-engine.js";

const QTYPES = ["choice", "score", "noul"];

/** y = W x + b for W [out, in] (row-major), b [out]. */
function linear(W: Float32Array, b: Float32Array | undefined, x: Float32Array, out: number, from = 0, cols = x.length, stride = x.length): Float32Array {
  const y = new Float32Array(out);
  for (let o = 0; o < out; o++) {
    let s = b ? b[o]! : 0;
    const r = o * stride + from;
    for (let k = 0; k < cols; k++) s += W[r + k]! * x[k]!;
    y[o] = s;
  }
  return y;
}

const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));
function erf(x: number) { // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}

function layerNorm(x: Float32Array, w: Float32Array, b: Float32Array, eps: number) {
  let m = 0, v = 0;
  for (const a of x) m += a;
  m /= x.length;
  for (const a of x) v += (a - m) * (a - m);
  const inv = 1 / Math.sqrt(v / x.length + eps);
  return x.map((a, i) => (a - m) * inv * w[i]! + b[i]!);
}

const dot = (a: Float32Array, b: Float32Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!; return s; };
const norm = (a: Float32Array) => Math.sqrt(dot(a, a));

/** An option's cached terms: its embedding normalized, LayerNorm(O), and the residual's first layer on it. */
interface OptionTerms { o: Float32Array; on: Float32Array; onNorm: number; r: Float32Array }

export class Eg2Head {
  private readonly W: number;
  private projS = new Map<string, Float32Array>();
  private projQ = new Map<string, Float32Array>();
  private terms = new Map<string, OptionTerms>();

  constructor(private readonly m: Eg2Manifest["config"], private readonly t: Record<string, Float32Array>) {
    this.W = m.head.width;
  }

  private cached<V>(map: Map<string, V>, key: string, make: () => V): V {
    let v = map.get(key);
    if (v === undefined) {
      if (map.size >= 4096) map.delete(map.keys().next().value!); // the oldest text first
      map.set(key, (v = make()));
    }
    return v;
  }

  /** The raw logit of every option: `s`, `q`, `opts` the normalized embeddings of the state, question and
   *  options, keyed by their text for the caches. */
  logits(type: string, s: [string, Float32Array], q: [string, Float32Array], opts: [string, Float32Array][]): number[] {
    const { t, W } = this, h = this.m.head;
    const S = this.cached(this.projS, s[0], () => linear(t["head.state_proj.weight"]!, t["head.state_proj.bias"], s[1], W));
    const Q = this.cached(this.projQ, q[0], () => linear(t["head.question_proj.weight"]!, t["head.question_proj.bias"], q[1], W));
    const fin = new Float32Array(3 * W);
    fin.set(S, 0); fin.set(Q, W);
    for (let i = 0; i < W; i++) fin[2 * W + i] = S[i]! * Q[i]!;
    const f1 = linear(t["head.field.0.weight"]!, t["head.field.0.bias"], fin, W).map(gelu);
    const f2 = linear(t["head.field.3.weight"]!, t["head.field.3.bias"], f1, W);
    const type0 = QTYPES.indexOf(type) * W, emb = t["head.type_emb.weight"]!;
    for (let i = 0; i < W; i++) f2[i]! += emb[type0 + i]!;
    const f = layerNorm(f2, t["head.field_norm.weight"]!, t["head.field_norm.bias"]!, h.ln_eps);
    const fNorm = norm(f);
    // the residual's first layer W·[f, Oₙ, f·Oₙ, |f − Oₙ|] = W_f f + W_o Oₙ + W_m (f·Oₙ) + W_a |f − Oₙ|: W_f f once
    // per question, W_o Oₙ once per option text
    const R0 = t["head.residual.0.weight"]!, rb = t["head.residual.0.bias"]!, R3 = t["head.residual.3.weight"]!;
    const rf = linear(R0, rb, f, W, 0, W, 4 * W);
    const anchor = new Float32Array(s[1].length);
    for (let i = 0; i < anchor.length; i++) anchor[i] = s[1][i]! + q[1][i]!;
    const an = Math.max(norm(anchor), 1e-12);
    return opts.map(([text, e]) => {
      const o = this.cached(this.terms, text, () => {
        const O = linear(t["head.option_proj.weight"]!, t["head.option_proj.bias"], e, W);
        const on = layerNorm(O, t["head.option_norm.weight"]!, t["head.option_norm.bias"]!, h.ln_eps);
        return { o: e.map((v) => v / Math.max(norm(e), 1e-12)), on, onNorm: norm(on), r: linear(R0, undefined, on, W, W, W, 4 * W) };
      });
      const prod = new Float32Array(W), diff = new Float32Array(W);
      for (let i = 0; i < W; i++) { prod[i] = f[i]! * o.on[i]!; diff[i] = Math.abs(f[i]! - o.on[i]!); }
      const rm = linear(R0, undefined, prod, W, 2 * W, W, 4 * W), ra = linear(R0, undefined, diff, W, 3 * W, W, 4 * W);
      let res = h.residual_bias;
      for (let i = 0; i < W; i++) res += R3[i]! * gelu(rf[i]! + o.r[i]! + rm[i]! + ra[i]!);
      const joint = h.joint_scale * dot(f, o.on) / Math.max(fNorm * o.onNorm, 1e-8);
      const prior = h.prior_scale * dot(o.o, anchor) / an;
      return prior + h.gate * (joint + res);
    });
  }
}
