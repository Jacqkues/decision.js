// WGSL kernels of the decision.js engines. Activations in f16, every reduction in f32. Many sequences are packed
// on one token axis: per-token metadata says which tokens each one may read (attention, convolution).

const HEAD = /* wgsl */ `enable f16;`;

/** x[T,H] = embedding rows (q4) of ids. */
export const GATHER_Q4 = /* wgsl */ `${HEAD}
struct P { T: u32, H: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> Wq: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f16>;
@group(0) @binding(4) var<storage, read_write> X: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.T * p.H) { return; }
  let t = i / p.H; let k = i % p.H; let row = ids[t];
  let word = Wq[row * (p.H / 8u) + k / 8u];
  let code = (word >> ((k % 8u) * 4u)) & 15u;
  X[i] = (f16(f32(code)) - f16(8.0)) * S[row * (p.H / 32u) + k / 32u];
}`;

/** One row per workgroup: optionally x += y (residual), then out = rmsnorm(x) * w. */
export const ADD_RMSNORM = /* wgsl */ `${HEAD}
struct P { T: u32, H: u32, add: u32, pad: u32, eps: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f16>;
@group(0) @binding(2) var<storage, read> Yr: array<f16>;
@group(0) @binding(3) var<storage, read> W: array<f16>;
@group(0) @binding(4) var<storage, read_write> O: array<f16>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let t = wg.x; let base = t * p.H;
  var ss = 0.0;
  for (var k = l.x; k < p.H; k += 256u) {
    var v = f32(X[base + k]);
    if (p.add == 1u) { v += f32(Yr[base + k]); X[base + k] = f16(v); }
    ss += v * v;
  }
  part[l.x] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { part[l.x] += part[l.x + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0] / f32(p.H) + p.eps);
  for (var k = l.x; k < p.H; k += 256u) { O[base + k] = f16(f32(X[base + k]) * inv * f32(W[k])); }
}`;

/** h[T,I] = silu(a[:, :I]) * a[:, I:] (a is the fused w1|w3 output). */
export const SILU_MUL = /* wgsl */ `${HEAD}
struct P { T: u32, I: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read_write> Hh: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.T * p.I) { return; }
  let t = i / p.I; let c = i % p.I;
  let a = f32(A[t * 2u * p.I + c]); let b = f32(A[t * 2u * p.I + p.I + c]);
  Hh[i] = f16(a / (1.0 + exp(-a)) * b);
}`;

/** LFM2 short conv over packed tokens: z = in_proj out [T, 3H] = (B | C | U); y = C * conv(B*U), 3 taps.
 *  Each token names the tokens its taps read (or -1): d1-3B is causal (t-2, t-1, t), each branch continuing
 *  its trunk; d1-omni is centred (t-1, t, t+1) within its sequence. */
export const SHORT_CONV = /* wgsl */ `${HEAD}
struct P { T: u32, H: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Z: array<f16>;
@group(0) @binding(2) var<storage, read> Wc: array<f16>;   // [H, 3]
@group(0) @binding(3) var<storage, read> taps: array<vec4<i32>>; // per token: the tokens read by taps 0, 1, 2
@group(0) @binding(4) var<storage, read_write> Yo: array<f16>;
fn bu(t: i32, c: u32) -> f32 {
  if (t < 0) { return 0.0; }
  let base = u32(t) * 3u * p.H;
  return f32(Z[base + c]) * f32(Z[base + 2u * p.H + c]);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.T * p.H) { return; }
  let t = i / p.H; let c = i % p.H;
  let tp = taps[t];
  let conv = f32(Wc[c * 3u]) * bu(tp.x, c) + f32(Wc[c * 3u + 1u]) * bu(tp.y, c) + f32(Wc[c * 3u + 2u]) * bu(tp.z, c);
  Yo[i] = f16(f32(Z[t * 3u * p.H + p.H + c]) * conv);
}`;

/** qkv [T, (32+8+8)*64] → per-head RMSNorm on q and k, half-split RoPE at each token's position; in place. */
export const QK_NORM_ROPE = /* wgsl */ `${HEAD}
struct P { T: u32, heads: u32, kv: u32, pad: u32, eps: f32, theta: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> QKV: array<f16>;
@group(0) @binding(2) var<storage, read> qn: array<f16>;
@group(0) @binding(3) var<storage, read> kn: array<f16>;
@group(0) @binding(4) var<storage, read> pos: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let idx = g.x + g.y * nw.x * 64u; /* 2-D past 65,535 workgroups */ let perTok = p.heads + p.kv; if (idx >= p.T * perTok) { return; }
  let t = idx / perTok; let h = idx % perTok;
  let isQ = h < p.heads;
  let base = t * (p.heads + 2u * p.kv) * 64u + h * 64u;
  var v: array<f32, 64>; var ss = 0.0;
  for (var d = 0u; d < 64u; d++) { v[d] = f32(QKV[base + d]); ss += v[d] * v[d]; }
  let inv = inverseSqrt(ss / 64.0 + p.eps);
  for (var d = 0u; d < 64u; d++) { v[d] = v[d] * inv * select(f32(kn[d]), f32(qn[d]), isQ); }
  let ps = f32(pos[t]);
  for (var d = 0u; d < 32u; d++) {
    let ang = ps * pow(p.theta, -f32(2u * d) / 64.0);
    let c = cos(ang); let s = sin(ang);
    let a = v[d]; let b = v[d + 32u];
    QKV[base + d] = f16(a * c - b * s);
    QKV[base + d + 32u] = f16(b * c + a * s);
  }
}`;

/** Attention over the forest. Token meta: (tree start, tree end, branch, offset): a trunk token (branch -1)
 *  sees earlier trunk tokens; a branch token sees the whole trunk and its own branch up to itself.
 *  bidir = 1: every token sees its whole [start, end) segment (d1-omni). Head dim 64. */
export const ATTENTION = /* wgsl */ `${HEAD}
struct P { T: u32, heads: u32, kv: u32, bidir: u32, scale: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> QKV: array<f16>;
@group(0) @binding(2) var<storage, read> tmeta: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read_write> O: array<f16>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let idx = g.x + g.y * nw.x * 64u; /* 2-D past 65,535 workgroups */ if (idx >= p.T * p.heads) { return; }
  let t = idx / p.heads; let h = idx % p.heads; let kvh = h / (p.heads / p.kv);
  let stride = (p.heads + 2u * p.kv) * 64u;
  let qb = t * stride + h * 64u;
  var q: array<f32, 64>;
  for (var d = 0u; d < 64u; d++) { q[d] = f32(QKV[qb + d]) * p.scale; }
  let mt = tmeta[t];
  var m = -1e30; var l = 0.0; var acc: array<f32, 64>;
  for (var k = mt.x; k < mt.y; k++) {
    let mk = tmeta[k];
    let ok = p.bidir == 1u || (mk.z == -1 && (mt.z != -1 || mk.w <= mt.w)) || (mk.z == mt.z && mt.z != -1 && mk.w <= mt.w);
    if (!ok) { continue; }
    let kb = u32(k) * stride + (p.heads + kvh) * 64u;
    var s = 0.0;
    for (var d = 0u; d < 64u; d++) { s += q[d] * f32(QKV[kb + d]); }
    let mn = max(m, s); let corr = exp(m - mn); let e = exp(s - mn);
    l = l * corr + e;
    let vb = kb + p.kv * 64u;
    for (var d = 0u; d < 64u; d++) { acc[d] = acc[d] * corr + e * f32(QKV[vb + d]); }
    m = mn;
  }
  for (var d = 0u; d < 64u; d++) { O[t * p.heads * 64u + h * 64u + d] = f16(acc[d] / l); }
}`;

/** Final norm at each leaf, then logits for the option tokens only (tied LM head = q4 embedding rows). */
export const LEAF_LOGITS = /* wgsl */ `${HEAD}
struct P { L: u32, H: u32, nIds: u32, pad: u32, eps: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> leaves: array<u32>;
@group(0) @binding(3) var<storage, read> W: array<f16>;      // final norm
@group(0) @binding(4) var<storage, read> ids: array<u32>;    // [L * nIds], token ids per leaf (padded)
@group(0) @binding(5) var<storage, read> Eq: array<u32>;
@group(0) @binding(6) var<storage, read> Es: array<f16>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
var<workgroup> part: array<f32, 256>;
var<workgroup> normed: array<f32, 2048>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let leaf = wg.x; let row = leaves[leaf] * p.H;
  var ss = 0.0;
  for (var k = l.x; k < p.H; k += 256u) { let v = f32(X[row + k]); ss += v * v; }
  part[l.x] = ss; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { part[l.x] += part[l.x + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0] / f32(p.H) + p.eps);
  workgroupBarrier();
  for (var k = l.x; k < p.H; k += 256u) { normed[k] = f32(X[row + k]) * inv * f32(W[k]); }
  workgroupBarrier();
  for (var j = 0u; j < p.nIds; j++) {
    let id = ids[leaf * p.nIds + j];
    var dot = 0.0;
    for (var k = l.x; k < p.H; k += 256u) {
      let code = (Eq[id * (p.H / 8u) + k / 8u] >> ((k % 8u) * 4u)) & 15u;
      dot += normed[k] * (f32(code) - 8.0) * f32(Es[id * (p.H / 32u) + k / 32u]);
    }
    part[l.x] = dot; workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { part[l.x] += part[l.x + s]; } workgroupBarrier(); }
    if (l.x == 0u) { out[leaf * p.nIds + j] = part[0]; }
    workgroupBarrier();
  }
}`;

// ---------- d1-omni decision head: pre-norm transformer layers with biases, then the marker scorer

/** Xo[t] = Xi[t] + E[idx[t]] (the question-type embedding, f16 rows). */
export const ADD_ROWS = /* wgsl */ `${HEAD}
struct P { T: u32, H: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Xi: array<f16>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> E: array<f16>;
@group(0) @binding(4) var<storage, read_write> Xo: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.T * p.H) { return; }
  let t = i / p.H; let k = i % p.H;
  Xo[i] = f16(f32(Xi[i]) + f32(E[idx[t] * p.H + k]));
}`;

/** Y[t, n] = act(Y[t, n] + b[n]) in place; act 0 = none, 1 = ReLU, 2 = GELU (erf). */
export const BIAS_ACT = /* wgsl */ `${HEAD}
struct P { T: u32, N: u32, act: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> Y: array<f16>;
@group(0) @binding(2) var<storage, read> b: array<f16>;
fn erf(x: f32) -> f32 { // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
  return select(-y, y, x >= 0.0);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.T * p.N) { return; }
  var v = f32(Y[i]) + f32(b[i % p.N]);
  if (p.act == 1u) { v = max(v, 0.0); }
  if (p.act == 2u) { v = 0.5 * v * (1.0 + erf(v * 0.70710678)); }
  Y[i] = f16(v);
}`;

/** One row per workgroup: optionally x += y (residual), then out = layernorm(x) * w + b. */
export const ADD_LAYERNORM = /* wgsl */ `${HEAD}
struct P { T: u32, H: u32, add: u32, pad: u32, eps: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f16>;
@group(0) @binding(2) var<storage, read> Yr: array<f16>;
@group(0) @binding(3) var<storage, read> W: array<f16>;
@group(0) @binding(4) var<storage, read> Bb: array<f16>;
@group(0) @binding(5) var<storage, read_write> O: array<f16>;
var<workgroup> s1: array<f32, 256>;
var<workgroup> s2: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let base = wg.x * p.H;
  var a = 0.0; var q = 0.0;
  for (var k = l.x; k < p.H; k += 256u) {
    var v = f32(X[base + k]);
    if (p.add == 1u) { v += f32(Yr[base + k]); X[base + k] = f16(v); }
    a += v; q += v * v;
  }
  s1[l.x] = a; s2[l.x] = q;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { s1[l.x] += s1[l.x + s]; s2[l.x] += s2[l.x + s]; } workgroupBarrier(); }
  let mean = s1[0] / f32(p.H);
  let inv = inverseSqrt(max(s2[0] / f32(p.H) - mean * mean, 0.0) + p.eps);
  for (var k = l.x; k < p.H; k += 256u) {
    O[base + k] = f16((f32(X[base + k]) - mean) * inv * f32(W[k]) + f32(Bb[k]));
  }
}`;

/** G[m] = X[rows[m]] (the hidden states at the option markers). */
export const GATHER_ROWS = /* wgsl */ `${HEAD}
struct P { M: u32, H: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read_write> G: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.M * p.H) { return; }
  G[i] = X[rows[i / p.H] * p.H + i % p.H];
}`;

/** out[m] = Y[m] · w + b, one workgroup per marker (the scorer's last Linear(H, 1)). */
export const SCORE = /* wgsl */ `${HEAD}
struct P { M: u32, H: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Y: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f16>;
@group(0) @binding(3) var<storage, read> b: array<f16>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  var d = 0.0;
  for (var k = l.x; k < p.H; k += 256u) { d += f32(Y[wg.x * p.H + k]) * f32(w[k]); }
  part[l.x] = d; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { part[l.x] += part[l.x + s]; } workgroupBarrier(); }
  if (l.x == 0u) { out[wg.x] = part[0] + f32(b[0]); }
}`;

// ---------- q8 (block 32, one byte per weight, implicit zero point 128, f16 scales)

const swap = (code: string, from: string, to: string) => {
  if (!code.includes(from)) throw new Error("q8 kernel: decode block not found");
  return code.replace(from, to);
};

/** GATHER_Q4 with q8 rows. */
export const GATHER_Q8 = swap(GATHER_Q4, `  let word = Wq[row * (p.H / 8u) + k / 8u];
  let code = (word >> ((k % 8u) * 4u)) & 15u;
  X[i] = (f16(f32(code)) - f16(8.0)) * S[row * (p.H / 32u) + k / 32u];`, `  let code = (Wq[(row * p.H + k) / 4u] >> ((k % 4u) * 8u)) & 255u;
  X[i] = (f16(f32(code)) - f16(128.0)) * S[row * (p.H / 32u) + k / 32u];`);

/** Y[M,N] = X[M,K] · Wᵀ on subgroup matrices (the GPU's matrix units, Chrome's
 *  chromium_experimental_subgroup_matrix). Derived from ONNX Runtime's WebGPU MatMulNBits "prefill_sgmat" kernel
 *  (Microsoft, MIT License; see NOTICE): f16 operands, whose products accumulate in f16
 *  on the matrix units, so partial sums are flushed into f32 registers every FLUSH values of K. Generated
 *  fully unrolled (arrays of subgroup matrices indexed in loops spill to memory on Metal): output tile
 *  TM×TN, each subgroup of 32 lanes owns SM×SN of it. */
export function matmulSgmat16({ tm, tn, sm, sn, bits }: { tm: number; tn: number; sm: number; sn: number; bits: 4 | 8 }) {
  const subs = (tm / sm) * (tn / sn), threads = subs * 32, rm = sm / 8, rn = sn / 8, per = (sm * sn) / 32;
  const L: string[] = [];
  for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) L.push(`  var c_${i}_${j} = subgroup_matrix_result<f16, 8, 8>();`);
  const decls = L.join("\n");
  const mma: string[] = [];
  for (let i = 0; i < rm; i++) mma.push(`      let a_${i} = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>, row_major>(&tile_A, (sy * ${sm}u + ${i * 8}u) * 32u + step, 32u);`);
  for (let j = 0; j < rn; j++) {
    mma.push(`      let b_${j} = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>, col_major>(&tile_B, (sx * ${sn}u + ${j * 8}u) * 32u + step, 32u);`);
    for (let i = 0; i < rm; i++) mma.push(`      c_${i}_${j} = subgroupMatrixMultiplyAccumulate(a_${i}, b_${j}, c_${i}_${j});`);
  }
  const store: string[] = [];
  for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) store.push(`      subgroupMatrixStore<row_major>(&scratch[sub], ${(i * rn + j) * 64}u, c_${i}_${j}, 8u); c_${i}_${j} = subgroup_matrix_result<f16, 8, 8>();`);
  const decode = bits === 4
    ? `let packed = b[((n * kblocks + kidx / 32u) * 16u + (col >> 1u)) >> 2u];
      for (var h = 0u; h < 8u; h++) { tile_B[row * 32u + col + h] = (f16(f32((packed >> (h * 4u)) & 15u)) - f16(8.0)) * scale; }`
    : `let w0 = (n * p.K + kidx + col) >> 2u;
      for (var h = 0u; h < 8u; h++) { tile_B[row * 32u + col + h] = (f16(f32((b[w0 + h / 4u] >> ((h % 4u) * 8u)) & 255u)) - f16(128.0)) * scale; }`;
  return /* wgsl */ `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);
diagnostic(off, subgroup_uniformity);
struct P { M: u32, N: u32, K: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f16>;
@group(0) @binding(4) var<storage, read_write> y: array<f16>;
const FLUSH: u32 = 128u;
var<workgroup> tile_A: array<f16, ${tm * 32}>;
var<workgroup> tile_B: array<f16, ${tn * 32}>;
var<workgroup> scratch: array<array<f16, ${sm * sn}>, ${subs}>;
// tile ${tm}x${tn}, subgroup ${sm}x${sn}
@compute @workgroup_size(${threads}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let a_base = wg.y * ${tm}u;
  let b_base = wg.x * ${tn}u;
  let sub = li / 32u;
  let sx = sub % ${tn / sn}u;
  let sy = sub / ${tn / sn}u;
  var acc: array<f32, ${per}>;
${decls}
  let kblocks = p.K / 32u;
  for (var kidx = 0u; kidx < p.K; kidx += 32u) {
    for (var ch = li; ch < ${tm * 4}u; ch += ${threads}u) { // A: 8 halves per chunk
      let row = ch / 4u; let col = (ch % 4u) * 8u; let g = a_base + row;
      let src = min(g, p.M - 1u) * p.K + kidx + col;
      for (var o = 0u; o < 8u; o++) { tile_A[row * 32u + col + o] = select(f16(0.0), a[src + o], g < p.M); }
    }
    for (var ch = li; ch < ${tn * 4}u; ch += ${threads}u) { // B: 8 weights per chunk
      let row = ch / 4u; let col = (ch % 4u) * 8u; let n = min(b_base + row, p.N - 1u);
      let scale = scales[n * kblocks + kidx / 32u];
      ${decode}
    }
    workgroupBarrier();
    for (var step = 0u; step < 32u; step += 8u) {
${mma.join("\n")}
    }
    workgroupBarrier();
    if ((kidx + 32u) % FLUSH == 0u || kidx + 32u >= p.K) {
${store.join("\n")}
      workgroupBarrier();
      for (var j = 0u; j < ${per}u; j++) { acc[j] += f32(scratch[sub][lane * ${per}u + j]); }
      workgroupBarrier();
    }
  }
  for (var j = 0u; j < ${per}u; j++) {
    let e = lane * ${per}u + j;
    let blk = e / 64u; let idx = e % 64u;
    let r = sy * ${sm}u + (blk / ${rn}u) * 8u + idx / 8u;
    let c = sx * ${sn}u + (blk % ${rn}u) * 8u + idx % 8u;
    let gm = a_base + r; let gn = b_base + c;
    if (gm < p.M && gn < p.N) { y[gm * p.N + gn] = f16(acc[j]); }
  }
}`;
}

/** The tiling used on matrix units: 64×64 tiles, 4 subgroups of 32×32 (16 accumulators each, 8 matrix
 *  loads per 16 products), the fastest of the shapes measured on Apple GPUs. */
export const MATMUL_Q4_SG32 = matmulSgmat16({ tm: 64, tn: 64, sm: 32, sn: 32, bits: 4 });
export const MATMUL_Q8_SG32 = matmulSgmat16({ tm: 64, tn: 64, sm: 32, sn: 32, bits: 8 });

/** Bidirectional attention by tiles (d1-omni): a workgroup takes 32 consecutive tokens of one head; keys
 *  and values of their segments stream through shared memory 32 at a time; 4 lanes share a query, 16 dims
 *  each, and add their partial scores with subgroup shuffles. Each token still sees only its own [start, end).
 *  Same bindings and uniform as ATTENTION; dispatch (ceil(T / 32), heads). */
export const ATTENTION_TILED = /* wgsl */ `enable f16;
enable subgroups;
diagnostic(off, subgroup_uniformity);
struct P { T: u32, heads: u32, kv: u32, bidir: u32, scale: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> QKV: array<f16>;
@group(0) @binding(2) var<storage, read> tmeta: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read_write> O: array<f16>;
const QT: u32 = 32u;
const KT: u32 = 32u;
var<workgroup> ks: array<f16, 2048>;
var<workgroup> vs: array<f16, 2048>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let h = wg.y; let t0 = wg.x * QT;
  let part = li % 4u; let t = min(t0 + li / 4u, p.T - 1u);
  let kvh = h / (p.heads / p.kv);
  let stride = (p.heads + 2u * p.kv) * 64u;
  var q: array<f32, 16>;
  for (var d = 0u; d < 16u; d++) { q[d] = f32(QKV[t * stride + h * 64u + part * 16u + d]) * p.scale; }
  let mt = tmeta[t];
  let kmin = u32(tmeta[t0].x); let kmax = u32(tmeta[min(t0 + QT - 1u, p.T - 1u)].y);
  var m = -1e30; var l = 0.0; var acc: array<f32, 16>;
  for (var kc = kmin; kc < kmax; kc += KT) {
    for (var e = li; e < KT * 64u; e += 128u) {
      let src = min(kc + e / 64u, p.T - 1u) * stride + (p.heads + kvh) * 64u + e % 64u;
      ks[e] = QKV[src]; vs[e] = QKV[src + p.kv * 64u];
    }
    workgroupBarrier();
    let n = min(KT, kmax - kc);
    for (var j = 0u; j < n; j++) {
      var s = 0.0;
      for (var d = 0u; d < 16u; d++) { s += q[d] * f32(ks[j * 64u + part * 16u + d]); }
      s += subgroupShuffleXor(s, 1u);
      s += subgroupShuffleXor(s, 2u);
      let k = i32(kc + j);
      if (k >= mt.x && k < mt.y) {
        let mn = max(m, s); let corr = exp(m - mn); let e = exp(s - mn);
        l = l * corr + e;
        for (var d = 0u; d < 16u; d++) { acc[d] = acc[d] * corr + e * f32(vs[j * 64u + part * 16u + d]); }
        m = mn;
      }
    }
    workgroupBarrier();
  }
  if (t0 + li / 4u < p.T) {
    for (var d = 0u; d < 16u; d++) { O[t * p.heads * 64u + h * 64u + part * 16u + d] = f16(acc[d] / l); }
  }
}`;

/** Generated portable matmul (no matrix units): Y[M,N] = X[M,K] · Wᵀ, q4 or q8 weights dequantized into
 *  shared memory. Tiles TM×TN×32 are stored k-major as vec4<f16> so each thread reads its RM rows and RN
 *  columns with vector loads and accumulates an RM×RN block in f32 registers. Fully unrolled: register
 *  arrays indexed in loops spill to memory on Metal. */
export function matmulPortable({ tm, tn, rm, rn, bits }: { tm: number; tn: number; rm: number; rn: number; bits: 4 | 8 }) {
  const tx = tn / rn, ty = tm / rm, threads = tx * ty, va = rm / 4, vb = rn / 4;
  const decl: string[] = [], loads: string[] = [], fma: string[] = [], store: string[] = [];
  for (let i = 0; i < rm; i++) for (let j = 0; j < vb; j++) decl.push(`  var c${i}_${j} = vec4<f32>();`);
  for (let i = 0; i < va; i++) loads.push(`      let a${i} = vec4<f32>(xs[kk * ${tm / 4}u + ty * ${va}u + ${i}u]);`);
  for (let j = 0; j < vb; j++) loads.push(`      let b${j} = vec4<f32>(ws[kk * ${tn / 4}u + tx * ${vb}u + ${j}u]);`);
  const comp = ["x", "y", "z", "w"];
  for (let i = 0; i < rm; i++) for (let j = 0; j < vb; j++) fma.push(`      c${i}_${j} = fma(vec4<f32>(a${i >> 2}.${comp[i & 3]}), b${j}, c${i}_${j});`);
  for (let i = 0; i < rm; i++) {
    store.push(`  { let gm = m0 + ty * ${rm}u + ${i}u; if (gm < p.M) {`);
    for (let j = 0; j < vb; j++) for (let c = 0; c < 4; c++) store.push(`    { let gn = n0 + tx * ${rn}u + ${j * 4 + c}u; if (gn < p.N) { Y[gm * p.N + gn] = f16(c${i}_${j}.${comp[c]}); } }`);
    store.push("  } }");
  }
  // every thread writes whole vec4s (4 columns at once): component writes would race
  const decode = bits === 4
    ? `for (var e = li; e < ${tn}u; e += ${threads}u) { // 4 columns × one word (8 values of K)
      let g = e / 4u; let wi = e % 4u;
      var w: array<u32, 4>; var sc: array<f16, 4>;
      for (var c = 0u; c < 4u; c++) {
        let gn = min(n0 + g * 4u + c, p.N - 1u);
        w[c] = Wq[gn * (p.K / 8u) + k0 / 8u + wi]; sc[c] = S[gn * (p.K / 32u) + k0 / 32u];
      }
      for (var j = 0u; j < 8u; j++) {
        let d = vec4<f16>(vec4<f32>((vec4<u32>(w[0], w[1], w[2], w[3]) >> vec4<u32>(j * 4u)) & vec4<u32>(15u))) - vec4<f16>(8.0);
        ws[(wi * 8u + j) * ${tn / 4}u + g] = d * vec4<f16>(sc[0], sc[1], sc[2], sc[3]);
      }
    }`
    : `for (var e = li; e < ${tn * 2}u; e += ${threads}u) { // 4 columns × one word (4 values of K)
      let g = e / 8u; let wi = e % 8u;
      var w: array<u32, 4>; var sc: array<f16, 4>;
      for (var c = 0u; c < 4u; c++) {
        let gn = min(n0 + g * 4u + c, p.N - 1u);
        w[c] = Wq[(gn * p.K + k0) / 4u + wi]; sc[c] = S[gn * (p.K / 32u) + k0 / 32u];
      }
      for (var j = 0u; j < 4u; j++) {
        let d = vec4<f16>(vec4<f32>((vec4<u32>(w[0], w[1], w[2], w[3]) >> vec4<u32>(j * 8u)) & vec4<u32>(255u))) - vec4<f16>(128.0);
        ws[(wi * 4u + j) * ${tn / 4}u + g] = d * vec4<f16>(sc[0], sc[1], sc[2], sc[3]);
      }
    }`;
  return /* wgsl */ `enable f16;
struct P { M: u32, N: u32, K: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> Wq: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f16>;
@group(0) @binding(4) var<storage, read_write> Y: array<f16>;
var<workgroup> xs: array<vec4<f16>, ${(32 * tm) / 4}>;
var<workgroup> ws: array<vec4<f16>, ${(32 * tn) / 4}>;
// portable tile ${tm}x${tn}
@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let m0 = wg.y * ${tm}u; let n0 = wg.x * ${tn}u;
  let tx = li % ${tx}u; let ty = li / ${tx}u;
${decl.join("\n")}
  for (var k0 = 0u; k0 < p.K; k0 += 32u) {
    for (var e = li; e < ${tm * 8}u; e += ${threads}u) { // 4 rows × one value of K
      let k = e % 32u; let g = e / 32u; let r = m0 + g * 4u;
      var v: vec4<f16>;
      for (var c = 0u; c < 4u; c++) { v[c] = select(f16(0.0), X[min(r + c, p.M - 1u) * p.K + k0 + k], r + c < p.M); }
      xs[k * ${tm / 4}u + g] = v;
    }
    ${decode}
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
${loads.join("\n")}
${fma.join("\n")}
    }
    workgroupBarrier();
  }
${store.join("\n")}
}`;
}

/** The portable tiling used without matrix units: 64×128 tiles, 8×8 outputs per thread (128 threads). */
export const MATMUL_PORTABLE_Q4 = matmulPortable({ tm: 64, tn: 128, rm: 8, rn: 8, bits: 4 });
export const MATMUL_PORTABLE_Q8 = matmulPortable({ tm: 64, tn: 128, rm: 8, rn: 8, bits: 8 });

// ---------- "direct" matmul on matrix units (after ML Drift's conv_wave_matrix): weights converted once to
// f16 in 8×8 blocks, so the kernel loads both operands straight from storage, with no shared-memory tiles

/** W16[((n/8)·(K/8) + k/8)·64 + (k%8)·8 + n%8] = dequantized W[n, k] (q4 or q8): the 8×8 block layout that
 *  MATMUL_DIRECT reads as right-hand subgroup matrices. One thread per weight. `swiglu`: W is a fused w1 | w3
 *  (N = 2I), reordered so each 64-column tile holds 32 columns of w1 then the same 32 of w3 (MATMUL_SWIGLU). */
export const tileWeights = (bits: 4 | 8, swiglu = false) => /* wgsl */ `enable f16;
struct P { N: u32, K: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<f16>;
@group(0) @binding(3) var<storage, read_write> W16: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.N * p.K) { return; }
  let n = i / p.K; let k = i % p.K;
  ${swiglu ? "let r = n % 64u; let src = select(p.N / 2u + (n / 64u) * 32u + r - 32u, (n / 64u) * 32u + r, r < 32u);" : "let src = n;"}
  ${bits === 4
    ? "let code = (Wq[(src * p.K + k) / 8u] >> ((k % 8u) * 4u)) & 15u; let v = (f16(f32(code)) - f16(8.0)) * S[src * (p.K / 32u) + k / 32u];"
    : "let code = (Wq[(src * p.K + k) / 4u] >> ((k % 4u) * 8u)) & 255u; let v = (f16(f32(code)) - f16(128.0)) * S[src * (p.K / 32u) + k / 32u];"}
  W16[((n / 8u) * (p.K / 8u) + k / 8u) * 64u + (k % 8u) * 8u + n % 8u] = v;
}`;

/** Y[M,N] = X[M,K] · Wᵀ with W from tileWeights: each subgroup loads its left matrices from X (row-major,
 *  stride K) and its right matrices from W16, multiplies on the matrix units, and flushes its f16 partial
 *  sums to f32 registers every FLUSH values of K. Output tile TM×TN, SM×SN per subgroup, fully unrolled.
 *  Rows of X past M are read (the activation buffers hold whole tiles) and their results dropped. */
export function matmulDirect({ tm, tn, sm, sn }: { tm: number; tn: number; sm: number; sn: number }, swiglu = false) {
  if (swiglu && (tm !== 32 || tn !== 64 || sm !== 32 || sn !== 64)) throw new Error("decision.js: the SwiGLU epilogue needs the 32×64 direct tile");
  const subs = (tm / sm) * (tn / sn), threads = subs * 32, rm = sm / 8, rn = sn / 8, per = (sm * sn) / 32;
  const decl: string[] = [], mma: string[] = [], store: string[] = [];
  for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) decl.push(`  var c_${i}_${j} = subgroup_matrix_result<f16, 8, 8>();`);
  for (let i = 0; i < rm; i++) mma.push(`    let a_${i} = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>, row_major>(&a, (row0 + ${i * 8}u) * p.K + k, p.K);`);
  for (let j = 0; j < rn; j++) {
    mma.push(`    let b_${j} = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>, row_major>(&w, (((col0 / 8u) + ${j}u) * kb + k / 8u) * 64u, 8u);`);
    for (let i = 0; i < rm; i++) mma.push(`    c_${i}_${j} = subgroupMatrixMultiplyAccumulate(a_${i}, b_${j}, c_${i}_${j});`);
  }
  for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) store.push(`      subgroupMatrixStore<row_major>(&scratch[sub], ${(i * rn + j) * 64}u, c_${i}_${j}, 8u); c_${i}_${j} = subgroup_matrix_result<f16, 8, 8>();`);
  return /* wgsl */ `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);
diagnostic(off, subgroup_uniformity);
struct P { M: u32, N: u32, K: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f16>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
const FLUSH: u32 = 128u;
var<workgroup> scratch: array<array<f16, ${sm * sn}>, ${subs}>;
// direct tile ${tm}x${tn}
@compute @workgroup_size(${threads}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let sub = li / 32u;
  let sx = sub % ${tn / sn}u; let sy = sub / ${tn / sn}u;
  let row0 = wg.y * ${tm}u + sy * ${sm}u;
  let col0 = wg.x * ${tn}u + sx * ${sn}u;
  let kb = p.K / 8u;
  var acc: array<f32, ${per}>;
${decl.join("\n")}
  for (var k = 0u; k < p.K; k += 8u) {
${mma.join("\n")}
    if ((k + 8u) % FLUSH == 0u || k + 8u >= p.K) {
${store.join("\n")}
      workgroupBarrier();
      for (var j = 0u; j < ${per}u; j++) { acc[j] += f32(scratch[sub][lane * ${per}u + j]); }
      workgroupBarrier();
    }
  }
${swiglu ? `  // SwiGLU epilogue: lane l holds 8×8 block l (block row l / 8, column l % 8); blocks 0-3 of a row are w1, 4-7 w3
  let bj = lane % 8u;
  for (var j = 0u; j < 64u; j++) {
    let up = subgroupShuffle(acc[j], (lane + 4u) % 32u);
    let gm = row0 + (lane / 8u) * 8u + j / 8u;
    if (bj < 4u && gm < p.M) {
      let a = f32(f16(acc[j])); let b = f32(f16(up)); // rounded as the f16 matmul output SILU_MUL reads
      y[gm * (p.N / 2u) + wg.x * 32u + bj * 8u + j % 8u] = f16(a / (1.0 + exp(-a)) * b);
    }
  }` : `  for (var j = 0u; j < ${per}u; j++) {
    let e = lane * ${per}u + j;
    let blk = e / 64u; let idx = e % 64u;
    let gm = row0 + (blk / ${rn}u) * 8u + idx / 8u;
    let gn = col0 + (blk % ${rn}u) * 8u + idx % 8u;
    if (gm < p.M && gn < p.N) { y[gm * p.N + gn] = f16(acc[j]); }
  }`}
}`;
}

/** matmulDirect for few rows: `ks` subgroups split K, each over its own K/ks slice of a TM×TN tile (one
 *  subgroup's worth), then add their f32 partial sums in shared memory. With few rows, a workgroup's walk along K
 *  is the whole latency (the GPU is mostly idle); split ks ways, it is ks times shorter. K must be a multiple
 *  of 8·ks. */
export function matmulSplitK({ tm, tn, ks }: { tm: number; tn: number; ks: number }) {
  const rm = tm / 8, rn = tn / 8, per = (tm * tn) / 32;
  const decl: string[] = [], mma: string[] = [], store: string[] = [];
  for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) decl.push(`  var c_${i}_${j} = subgroup_matrix_result<f16, 8, 8>();`);
  for (let i = 0; i < rm; i++) mma.push(`    let a_${i} = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>, row_major>(&a, (row0 + ${i * 8}u) * p.K + k, p.K);`);
  for (let j = 0; j < rn; j++) {
    mma.push(`    let b_${j} = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>, row_major>(&w, (((col0 / 8u) + ${j}u) * kb + k / 8u) * 64u, 8u);`);
    for (let i = 0; i < rm; i++) mma.push(`    c_${i}_${j} = subgroupMatrixMultiplyAccumulate(a_${i}, b_${j}, c_${i}_${j});`);
  }
  for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) store.push(`      subgroupMatrixStore<row_major>(&scratch[sub], ${(i * rn + j) * 64}u, c_${i}_${j}, 8u); c_${i}_${j} = subgroup_matrix_result<f16, 8, 8>();`);
  return /* wgsl */ `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);
diagnostic(off, subgroup_uniformity);
struct P { M: u32, N: u32, K: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f16>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
const FLUSH: u32 = 128u;
var<workgroup> scratch: array<array<f16, ${tm * tn}>, ${ks}>;
var<workgroup> red: array<array<f32, ${tm * tn}>, ${ks}>;
// split-K tile ${tm}x${tn}/${ks}
@compute @workgroup_size(${ks * 32}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let sub = li / 32u;
  let row0 = wg.y * ${tm}u;
  let col0 = wg.x * ${tn}u;
  let kb = p.K / 8u;
  let span = p.K / ${ks}u; let kbeg = sub * span;
  var acc: array<f32, ${per}>;
${decl.join("\n")}
  for (var i = 0u; i < span; i += 8u) { // the same steps in every subgroup: the barriers stay uniform
    let k = kbeg + i;
${mma.join("\n")}
    if ((i + 8u) % FLUSH == 0u || i + 8u >= span) {
${store.join("\n")}
      workgroupBarrier();
      for (var j = 0u; j < ${per}u; j++) { acc[j] += f32(scratch[sub][lane * ${per}u + j]); }
      workgroupBarrier();
    }
  }
  for (var j = 0u; j < ${per}u; j++) { red[sub][lane * ${per}u + j] = acc[j]; }
  workgroupBarrier();
  for (var e = li; e < ${tm * tn}u; e += ${ks * 32}u) {
    var s = 0.0;
    for (var q = 0u; q < ${ks}u; q++) { s += red[q][e]; }
    let blk = e / 64u; let idx = e % 64u;
    let gm = row0 + (blk / ${rn}u) * 8u + idx / 8u;
    let gn = col0 + (blk % ${rn}u) * 8u + idx % 8u;
    if (gm < p.M && gn < p.N) { y[gm * p.N + gn] = f16(s); }
  }
}`;
}

/** The direct tiling: one subgroup per workgroup computing 32×64 outputs (32 accumulators): with no shared
 *  tiles, bigger workgroups only add barriers. Fastest of the shapes measured on Apple GPUs (within 3%). */
export const DIRECT_TILE = { tm: 32, tn: 64, sm: 32, sn: 64 };
export const MATMUL_DIRECT = matmulDirect(DIRECT_TILE);
/** H[M, I] = silu(X · W1ᵀ) · (X · W3ᵀ) with w1 | w3 from tileWeights(bits, true): the direct matmul with
 *  SILU_MUL in its epilogue (the gate and up halves of a row block sit in lanes l and l + 4). */
export const MATMUL_SWIGLU = matmulDirect(DIRECT_TILE, true);

// ---------- portable f16 matmul (no matrix units): weights converted once to f16, k-major in groups of 4
// output columns, so each thread reads its columns as vec4 straight from storage, with no dequantization

/** W4[k·(N/4) + n/4] = (W[n, k], W[n+1, k], W[n+2, k], W[n+3, k]) dequantized (q4 or q8). One thread per vec4. */
export const tileWeightsPortable = (bits: 4 | 8) => /* wgsl */ `enable f16;
struct P { N: u32, K: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<f16>;
@group(0) @binding(3) var<storage, read_write> W4: array<vec4<f16>>;
fn wv(n: u32, k: u32) -> f16 {
  ${bits === 4
    ? "let code = (Wq[(n * p.K + k) / 8u] >> ((k % 8u) * 4u)) & 15u; return (f16(f32(code)) - f16(8.0)) * S[n * (p.K / 32u) + k / 32u];"
    : "let code = (Wq[(n * p.K + k) / 4u] >> ((k % 4u) * 8u)) & 255u; return (f16(f32(code)) - f16(128.0)) * S[n * (p.K / 32u) + k / 32u];"}
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.K * (p.N / 4u)) { return; }
  let k = i / (p.N / 4u); let n = (i % (p.N / 4u)) * 4u;
  W4[i] = vec4<f16>(wv(n, k), wv(n + 1u, k), wv(n + 2u, k), wv(n + 3u, k));
}`;

/** Y[M,N] = X[M,K] · Wᵀ with W from tileWeightsPortable (N a multiple of TN). Each thread accumulates an
 *  RM×RN block in f32: its RN columns are read as vec4 from storage; its RM rows come from a k-major
 *  shared tile (aShared) or straight from storage as vec4 along K (no shared memory, no barrier). Unrolled. */
export function matmulPortableF16({ tm, tn, rm, rn, aShared }: { tm: number; tn: number; rm: number; rn: number; aShared: boolean }) {
  const tx = tn / rn, ty = tm / rm, threads = tx * ty, va = rm / 4, vb = rn / 4;
  const comp = ["x", "y", "z", "w"];
  const decl: string[] = [], body: string[] = [], store: string[] = [];
  for (let i = 0; i < rm; i++) for (let j = 0; j < vb; j++) decl.push(`  var c${i}_${j} = vec4<f32>();`);
  for (let i = 0; i < rm; i++) {
    store.push(`  { let gm = m0 + ty * ${rm}u + ${i}u; if (gm < p.M) {`);
    for (let j = 0; j < vb; j++) store.push(`    Y4[(gm * p.N + n0 + tx * ${rn}u) / 4u + ${j}u] = vec4<f16>(c${i}_${j});`);
    store.push("  } }");
  }
  const bLoads = (k: string) => Array.from({ length: vb }, (_, j) => `      let b${j} = vec4<f32>(W4[(${k}) * nq + cb + ${j}u]);`).join("\n");
  let main: string;
  if (aShared) {
    const fma: string[] = [];
    for (let i = 0; i < rm; i++) for (let j = 0; j < vb; j++) fma.push(`      c${i}_${j} = fma(vec4<f32>(a${i >> 2}.${comp[i & 3]}), b${j}, c${i}_${j});`);
    main = `  for (var k0 = 0u; k0 < p.K; k0 += 32u) {
    for (var e = li; e < ${tm * 8}u; e += ${threads}u) { // 4 rows × one value of K
      let k = e % 32u; let g = e / 32u; let r = m0 + g * 4u;
      var v: vec4<f16>;
      for (var c = 0u; c < 4u; c++) { v[c] = X[min(r + c, p.M - 1u) * p.K + k0 + k]; }
      xs[k * ${tm / 4}u + g] = v;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
${Array.from({ length: va }, (_, i) => `      let a${i} = vec4<f32>(xs[kk * ${tm / 4}u + ty * ${va}u + ${i}u]);`).join("\n")}
${bLoads("k0 + kk")}
${fma.join("\n")}
    }
    workgroupBarrier();
  }`;
  } else {
    const rows = Array.from({ length: rm }, (_, i) => `  let r${i} = min(m0 + ty * ${rm}u + ${i}u, p.M - 1u) * (p.K / 4u);`).join("\n");
    const inner: string[] = [];
    for (let i = 0; i < rm; i++) inner.push(`    let a${i} = vec4<f32>(X4[r${i} + k4]);`);
    for (let c = 0; c < 4; c++) {
      inner.push(bLoads(`k4 * 4u + ${c}u`).replaceAll("let b", `let b${c}_`).replace(/^ {6}/gm, "    "));
      for (let i = 0; i < rm; i++) for (let j = 0; j < vb; j++) inner.push(`    c${i}_${j} = fma(vec4<f32>(a${i}.${comp[c]}), b${c}_${j}, c${i}_${j});`);
    }
    main = `${rows}
  for (var k4 = 0u; k4 < p.K / 4u; k4++) {
${inner.join("\n")}
  }`;
  }
  return /* wgsl */ `enable f16;
struct P { M: u32, N: u32, K: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ${aShared ? "X: array<f16>" : "X4: array<vec4<f16>>"};
@group(0) @binding(2) var<storage, read> W4: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> Y4: array<vec4<f16>>;
${aShared ? `var<workgroup> xs: array<vec4<f16>, ${(32 * tm) / 4}>;` : ""}
// portable f16 tile ${tm}x${tn}
@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let m0 = wg.y * ${tm}u; let n0 = wg.x * ${tn}u;
  let tx = li % ${tx}u; let ty = li / ${tx}u;
  let nq = p.N / 4u; let cb = (n0 + tx * ${rn}u) / 4u;
${decl.join("\n")}
${main}
${store.join("\n")}
}`;
}

/** The portable f16 tiling: 64×128 tiles, 4×8 outputs per thread (256 threads), activations read straight
 *  from storage (no shared memory, no barrier): ~15% faster than staging them in shared memory on Apple GPUs. */
export const PORTABLE_F16_TILE = { tm: 64, tn: 128, rm: 4, rn: 8, aShared: false };
export const MATMUL_PORTABLE_F16 = matmulPortableF16(PORTABLE_F16_TILE);

