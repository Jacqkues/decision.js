// Kernels of EmbeddingGemma 2's text encoder: Gemma's sandwich RMSNorms around each block, q/k/v RMSNorm then
// RoPE, bidirectional attention with 256- or 512-dim heads (GQA, a window on the sliding layers), GELU (tanh)
// gating for the MLP and the per-layer embeddings, and the mean pooling with the embedding projection.
// Token metadata as everywhere: per token (sequence start, sequence end, -1, position).

const HEAD = /* wgsl */ `enable f16;`;
/** A flat index over a dispatch that may be 2-D (more than 65,535 workgroups). */
const WG = `let wid = wg.x + wg.y * nw.x;`;
const GELU = /* wgsl */ `fn gelu(x: f32) -> f32 { // tanh approximation (gelu_pytorch_tanh); tanh saturates long before 15
  let u = clamp(0.7978845608 * (x + 0.044715 * x * x * x), -15.0, 15.0);
  let e = exp(2.0 * u);
  return 0.5 * x * (1.0 + (e - 1.0) / (e + 1.0));
}`;

/** One row of H = 512 per workgroup (2 values per thread): x = (x + rmsnorm(y) · w1) · scale, then, when `norm2`,
 *  out = rmsnorm(x) · w2. Gemma's post-norm on a block's output, the residual add, the layer scalar, and the next
 *  block's pre-norm in one pass. */
export const SANDWICH = /* wgsl */ `${HEAD}
struct P { T: u32, H: u32, norm2: u32, pad: u32, eps: f32, scale: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f16>;
@group(0) @binding(2) var<storage, read> Y: array<f16>;
@group(0) @binding(3) var<storage, read> W1: array<f16>;
@group(0) @binding(4) var<storage, read> W2: array<f16>;
@group(0) @binding(5) var<storage, read_write> O: array<f16>;
var<workgroup> part: array<f32, 256>;
fn total(v: f32, l: u32) -> f32 {
  workgroupBarrier();
  part[l] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l < s) { part[l] += part[l + s]; } workgroupBarrier(); }
  return part[0];
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) l: u32) {
  ${WG} if (wid >= p.T) { return; }
  let base = wid * p.H;
  let y0 = f32(Y[base + l]); let y1 = f32(Y[base + l + 256u]);
  let iy = inverseSqrt(total(y0 * y0 + y1 * y1, l) / f32(p.H) + p.eps);
  let x0 = (f32(X[base + l]) + y0 * iy * f32(W1[l])) * p.scale;
  let x1 = (f32(X[base + l + 256u]) + y1 * iy * f32(W1[l + 256u])) * p.scale;
  X[base + l] = f16(x0); X[base + l + 256u] = f16(x1);
  if (p.norm2 == 1u) {
    let a = f32(f16(x0)); let b = f32(f16(x1)); // the stored residual, as the next block reads it
    let ix = inverseSqrt(total(a * a + b * b, l) / f32(p.H) + p.eps);
    O[base + l] = f16(a * ix * f32(W2[l])); O[base + l + 256u] = f16(b * ix * f32(W2[l + 256u]));
  }
}`;

/** One (token, head) per workgroup of 64 threads over the fused [q heads | k heads | v heads] rows of head dim D
 *  (256 or 512): RMSNorm (weighted for q and k, plain for v), then RoPE on q and k (rotate-half over all D dims,
 *  at the token's position); in place. */
export const QKV_NORM_ROPE = /* wgsl */ `${HEAD}
struct P { T: u32, heads: u32, kv: u32, D: u32, eps: f32, theta: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> Z: array<f16>;
@group(0) @binding(2) var<storage, read> qn: array<f16>;
@group(0) @binding(3) var<storage, read> kn: array<f16>;
@group(0) @binding(4) var<storage, read> tmeta: array<vec4<i32>>;
var<workgroup> part: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) l: u32) {
  ${WG} let per = p.heads + 2u * p.kv; if (wid >= p.T * per) { return; }
  let t = wid / per; let h = wid % per;
  let base = t * per * p.D + h * p.D;
  let n = p.D / 64u; // values per thread: 4 or 8, interleaved so that d and d + D/2 sit in the same thread
  var v: array<f32, 8>;
  var ss = 0.0;
  for (var j = 0u; j < n; j++) {
    let d = select(l + (j - n / 2u) * 64u + p.D / 2u, l + j * 64u, j < n / 2u);
    v[j] = f32(Z[base + d]); ss += v[j] * v[j];
  }
  part[l] = ss; workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (l < s) { part[l] += part[l + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0] / f32(p.D) + p.eps);
  let isQ = h < p.heads; let isK = !isQ && h < p.heads + p.kv;
  for (var j = 0u; j < n; j++) {
    let d = select(l + (j - n / 2u) * 64u + p.D / 2u, l + j * 64u, j < n / 2u);
    v[j] *= inv;
    if (isQ) { v[j] *= f32(qn[d]); } else if (isK) { v[j] *= f32(kn[d]); }
  }
  if (isQ || isK) {
    let ps = f32(tmeta[t].w);
    for (var j = 0u; j < n / 2u; j++) {
      let d = l + j * 64u;
      let ang = ps * (1.0 / pow(p.theta, f32(2u * d) / f32(p.D)));
      let c = cos(ang); let s = sin(ang);
      let a = v[j]; let b = v[j + n / 2u];
      v[j] = a * c - b * s; v[j + n / 2u] = b * c + a * s;
    }
  }
  for (var j = 0u; j < n; j++) {
    let d = select(l + (j - n / 2u) * 64u + p.D / 2u, l + j * 64u, j < n / 2u);
    Z[base + d] = f16(v[j]);
  }
}`;

/** Bidirectional attention with D-dim heads (256 or 512) over each token's key range [x, y) (its sequence, or its
 *  window in it), GQA, scale `scale`: `qt` queries × `g` query heads of one key/value head per workgroup (so each
 *  key/value tile is loaded once for the g heads), 16 lanes per (query, head) (D/16 dims each, read as vec4),
 *  keys and values through shared memory 4096 values at a time; each tile's scores come first (independent of
 *  each other), then one online-softmax update per tile. Rows of Z: [q heads | k heads | v heads].
 *  Dispatch (ceil(T / qt), heads / g). */
export function attention(D: 256 | 512, qt: number, g: number): string {
  const threads = qt * g * 16, per = D / 64, KT = 4096 / D; // per: vec4s per lane
  return /* wgsl */ `${HEAD}
enable subgroups;
diagnostic(off, subgroup_uniformity);
struct P { T: u32, heads: u32, kv: u32, pad: u32, scale: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Z: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> rng: array<vec4<i32>>;    // per token: keys [x, y)
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<f16>>; // [T, heads·D]
const D4: u32 = ${D / 4}u;
const QT: u32 = ${qt}u;
const KT: u32 = ${KT}u;
var<workgroup> ks: array<vec4<f16>, 1024>;
var<workgroup> vs: array<vec4<f16>, 1024>;
@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let stride = (p.heads + 2u * p.kv) * D4;
  let h = wg.y * ${g}u + (li / 16u) % ${g}u; let t0 = wg.x * QT;
  let part = li % 16u; let t = min(t0 + li / ${16 * g}u, p.T - 1u);
  let kvh = h / (p.heads / p.kv);
  let koff = p.heads * D4 + kvh * D4; let voff = (p.heads + p.kv) * D4 + kvh * D4;
  var q: array<vec4<f32>, ${per}>;
  for (var d = 0u; d < ${per}u; d++) { q[d] = vec4<f32>(Z[t * stride + h * D4 + part * ${per}u + d]) * p.scale; }
  let mt = rng[t];
  var kmin = rng[t0].x; var kmax = rng[t0].y;
  for (var i = 1u; i < QT; i++) { let r = rng[min(t0 + i, p.T - 1u)]; kmin = min(kmin, r.x); kmax = max(kmax, r.y); }
  var m = -1e30; var l = 0.0; var acc: array<vec4<f32>, ${per}>;
  for (var kc = u32(kmin); kc < u32(kmax); kc += KT) {
    for (var e = li; e < KT * D4; e += ${threads}u) {
      let row = min(kc + e / D4, p.T - 1u) * stride;
      ks[e] = Z[row + koff + e % D4]; vs[e] = Z[row + voff + e % D4];
    }
    workgroupBarrier();
    var sc: array<f32, ${KT}>;
    var mb = m;
    for (var j = 0u; j < KT; j++) {
      var s = 0.0;
      for (var d = 0u; d < ${per}u; d++) { s += dot(q[d], vec4<f32>(ks[j * D4 + part * ${per}u + d])); }
      s += subgroupShuffleXor(s, 1u); s += subgroupShuffleXor(s, 2u); s += subgroupShuffleXor(s, 4u); s += subgroupShuffleXor(s, 8u);
      let k = i32(kc + j);
      sc[j] = select(-1e30, s, k >= mt.x && k < mt.y && k < kmax);
      mb = max(mb, sc[j]);
    }
    let corr = exp(m - mb);
    l *= corr;
    for (var d = 0u; d < ${per}u; d++) { acc[d] *= corr; }
    for (var j = 0u; j < KT; j++) {
      let ex = select(0.0, exp(sc[j] - mb), sc[j] > -1e29);
      l += ex;
      for (var d = 0u; d < ${per}u; d++) { acc[d] += ex * vec4<f32>(vs[j * D4 + part * ${per}u + d]); }
    }
    m = mb;
    workgroupBarrier();
  }
  if (t0 + li / ${16 * g}u < p.T) {
    for (var d = 0u; d < ${per}u; d++) { Y[t * p.heads * D4 + h * D4 + part * ${per}u + d] = vec4<f16>(acc[d] / l); }
  }
}`;
}

/** h[T, I] = gelu(a[:, :I]) · a[:, I:] (a = the fused gate | up output). */
export const GELU_MUL = /* wgsl */ `${HEAD}
struct P { T: u32, I: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read_write> Hh: array<f16>;
${GELU}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.T * p.I) { return; }
  let t = i / p.I; let c = i % p.I;
  Hh[i] = f16(gelu(f32(A[t * 2u * p.I + c])) * f32(A[t * 2u * p.I + p.I + c]));
}`;

/** One row of 512 per workgroup: out = gelu(gate) · rmsnorm(e) · w, the per-layer embedding (e, this layer's
 *  slice of the embedding's projection, RMS-normed) gating the layer's own projection of x. */
export const PLE_MIX = /* wgsl */ `${HEAD}
struct P { T: u32, H: u32, pad0: u32, pad1: u32, eps: f32, pad2: f32, pad3: f32, pad4: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> G: array<f16>;
@group(0) @binding(2) var<storage, read> E: array<f16>;
@group(0) @binding(3) var<storage, read> W: array<f16>;
@group(0) @binding(4) var<storage, read_write> O: array<f16>;
var<workgroup> part: array<f32, 256>;
${GELU}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) l: u32) {
  ${WG} if (wid >= p.T) { return; }
  let base = wid * p.H;
  let e0 = f32(E[base + l]); let e1 = f32(E[base + l + 256u]);
  part[l] = e0 * e0 + e1 * e1; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l < s) { part[l] += part[l + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0] / f32(p.H) + p.eps);
  O[base + l] = f16(gelu(f32(G[base + l])) * e0 * inv * f32(W[l]));
  O[base + l + 256u] = f16(gelu(f32(G[base + l + 256u])) * e1 * inv * f32(W[l + 256u]));
}`;

/** One sequence per workgroup: the mean of its rows of X [T, 512] (f32), then out[s] = mean · Wᵀ (W [N, 512] f16,
 *  the embedding projection), in f32. Projecting after the mean equals the model's projection of every token. */
export const POOL_PROJECT = /* wgsl */ `${HEAD}
struct P { S: u32, H: u32, N: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> W: array<f16>;
@group(0) @binding(3) var<storage, read> seqs: array<vec2<u32>>;  // [start, end)
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> mean: array<f32, 512>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let s = wg.x; let r = seqs[s];
  var a = 0.0; var b = 0.0;
  for (var t = r.x; t < r.y; t++) { a += f32(X[t * p.H + l]); b += f32(X[t * p.H + l + 256u]); }
  let n = f32(r.y - r.x);
  mean[l] = a / n; mean[l + 256u] = b / n;
  workgroupBarrier();
  for (var o = l; o < p.N; o += 256u) {
    var d = 0.0;
    for (var k = 0u; k < p.H; k += 4u) {
      d += dot(vec4<f32>(mean[k], mean[k + 1u], mean[k + 2u], mean[k + 3u]),
               vec4<f32>(f32(W[o * p.H + k]), f32(W[o * p.H + k + 1u]), f32(W[o * p.H + k + 2u]), f32(W[o * p.H + k + 3u])));
    }
    out[s * p.N + o] = d;
  }
}`;

