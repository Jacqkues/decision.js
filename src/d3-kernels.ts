// Kernels of d3-lite's Qwen3.5 backbone: the Gated DeltaNet mixer (causal 4-tap conv + SiLU, q/k L2 norm and
// gates, the recurrence, the gated RMSNorm, in one kernel), the gated attention with 256-dim heads, and the code readout.
// Token metadata as everywhere: per token (sequence start, sequence end, -1, position).

const HEAD = /* wgsl */ `enable f16;`;
/** A flat index over a dispatch that may be 2-D (more than 65,535 workgroups). */
const FLAT = (size: number) => `let i = g.x + g.y * nw.x * ${size}u;`;

/** The whole Gated DeltaNet mixer after the in-projection, one workgroup per (sequence, head), `lanes` lanes per
 *  value row (128 rows, 128 / lanes f32 state floats per lane). Per chunk of 16 tokens:
 *  1. the causal 4-tap conv + SiLU of the chunk's q, k and v channels (within the sequence; rounded to f16 as
 *     a stored activation would be), q and k L2-normalized in f32 (q also scaled by dk^-1/2), and the gates:
 *     decay exp(g), g = -exp(A_log) · softplus(a + dt_bias), β = sigmoid(b), all into shared memory;
 *  2. the gated delta rule, after MLX's packed kernel: per token S ← exp(g)·S; δ = β(v − Sᵀk); S ← S + k δᵀ;
 *     o = Sᵀq (o replaces v in shared memory);
 *  3. the gated RMSNorm, y = rmsnorm(o) · w · silu(z), written out.
 *  A workgroup per head instead of per group of rows halves the parallelism of the recurrence, but saves the
 *  conv, prep and norm passes: 30% less GPU time than those 4 kernels on Apple M5 Pro (fewer rows: slower).
 *  Shared q / k are interleaved (vec4 j of part p at j·lanes + p) so the lanes of a row hit distinct banks.
 *  A sequence is span a then span b: with a shared state, a is the trunk (scanned again by every branch, its
 *  outputs written by the first one only) and b the question; otherwise a is the whole sequence and b empty.
 *  Dispatch (1, sequences · H). Fully unrolled (indexed register arrays spill on Metal). */
export function gdnLayer(lanes: 2 | 4, form: "mlx" | "ort" = "ort"): string {
  const J = 32 / lanes, e = Array.from({ length: J }, (_, j) => j);
  const shuffles = (v: string) => Array.from({ length: Math.log2(lanes) }, (_, i) => `${v} += subgroupShuffleXor(${v}, ${1 << i}u);`).join(" ");
  const decay = e.map((j) => `      s${j} *= dec; kv += dot(s${j}, k${j});`).join("\n");
  const upd = e.map((j) => `      s${j} += k${j} * delta; o += dot(s${j}, q${j});`).join("\n");
  // ORT form: r = Sᵀk and p = Sᵀq on the old state in one pass (one reduction phase per token), then
  // δ = β(v − dec·r), o = dec·p + (k·q)·δ, and the update S ← dec·S + k δᵀ with nothing left to reduce
  const dots = e.map((j) => `      r += dot(s${j}, k${j}); pq += dot(s${j}, q${j}); kq += dot(k${j}, q${j});`).join("\n");
  const step = e.map((j) => `      s${j} = s${j} * dec + k${j} * delta;`).join("\n");
  const loads = e.map((j) => `      let k${j} = ks[c * 32u + ${j * lanes}u + part]; let q${j} = qs[c * 32u + ${j * lanes}u + part];`).join("\n");
  const threads = 128 * lanes, subs = threads / 32;
  return /* wgsl */ `${HEAD}
enable subgroups;
diagnostic(off, subgroup_uniformity);
struct P { nseq: u32, H: u32, stride: u32, qkv: u32, eps: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> seqs: array<vec4<u32>>;   // (a start, a length | write a << 31, b start, b length)
@group(0) @binding(2) var<storage, read> Z: array<f16>;            // in-projection rows: q | k | v | z | b | a
@group(0) @binding(3) var<storage, read> W: array<f16>;            // conv taps [q | k | v channels, 4]
@group(0) @binding(4) var<storage, read> Alog: array<f16>;
@group(0) @binding(5) var<storage, read> dtb: array<f16>;
@group(0) @binding(6) var<storage, read> gw: array<f16>;           // gated norm weight [128]
@group(0) @binding(7) var<storage, read_write> Y: array<f16>;      // [T, H·128]
const C: u32 = 16u;
var<workgroup> span: vec4<u32>;
var<workgroup> qs: array<vec4<f32>, ${16 * 32}>;
var<workgroup> ks: array<vec4<f32>, ${16 * 32}>;
var<workgroup> vs: array<f32, ${16 * 128}>;                        // v, then o
var<workgroup> gs: array<vec4<f32>, 16>;                           // (decay, beta, k·q, -)
/** The token at index v of a sequence made of span a (a shared trunk, or the whole sequence) then span b. */
fn phys(v: u32, sl: vec4<u32>, alen: u32) -> u32 { return select(sl.z + v - alen, sl.x + v, v < alen); }
/** The input of channel ch (4 channels: raw4) at sequence index v - back, or 0 before the sequence start. */
fn raw1(v: u32, back: u32, sl: vec4<u32>, alen: u32, ch: u32) -> f32 {
  if (v < back) { return 0.0; }
  return f32(Z[phys(v - back, sl, alen) * p.stride + ch]);
}
fn raw4(v: u32, back: u32, sl: vec4<u32>, alen: u32, ch: u32) -> vec4<f32> {
  if (v < back) { return vec4<f32>(); }
  let b = phys(v - back, sl, alen) * p.stride + ch;
  return vec4<f32>(f32(Z[b]), f32(Z[b + 1u]), f32(Z[b + 2u]), f32(Z[b + 3u]));
}
/** Tap j of channels ch … ch + 3. */
fn taps4(ch: u32, j: u32) -> vec4<f32> {
  return vec4<f32>(f32(W[ch * 4u + j]), f32(W[(ch + 1u) * 4u + j]), f32(W[(ch + 2u) * 4u + j]), f32(W[(ch + 3u) * 4u + j]));
}
/** SiLU, rounded to f16 as the stored conv output it replaces. */
fn silu16(x: vec4<f32>) -> vec4<f32> { return vec4<f32>(vec4<f16>(x / (1.0 + exp(-x)))); }
@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let sq = wg.y / p.H; let h = wg.y % p.H;
  let row = li / ${lanes}u; let part = li % ${lanes}u; let sub = li / 32u;
  if (li == 0u) { span = seqs[sq]; }
  let sl = workgroupUniformLoad(&span);
  let alen = sl.y & 0x7fffffffu; let writeA = (sl.y >> 31u) == 1u; let end = alen + sl.w;
  let QK = p.H * 128u;
${e.map((j) => `  var s${j} = vec4<f32>();`).join("\n")}
  for (var t0 = 0u; t0 < end; t0 += C) { // t0, c: indices in the sequence; t: the token
    let n = min(C, end - t0);
    // 1. conv, norms and gates: each thread slides over a few consecutive tokens with the 3 previous inputs in
    //    registers, so every input is read about once. Half the subgroups: q and k (4 channels per lane, the
    //    norms by subgroup reduction); the other half: v (one channel per thread) and the gates.
    if (sub < ${subs / 2}u) {
      let isK = (sub % 2u) == 1u; let g0 = (sub / 2u) * ${16 / (subs / 4)}u;
      let ch = select(0u, QK, isK) + h * 128u + lane * 4u;
      let w0 = taps4(ch, 0u); let w1 = taps4(ch, 1u); let w2 = taps4(ch, 2u); let w3 = taps4(ch, 3u);
      var xm3 = raw4(t0 + g0, 3u, sl, alen, ch); var xm2 = raw4(t0 + g0, 2u, sl, alen, ch); var xm1 = raw4(t0 + g0, 1u, sl, alen, ch);
      for (var c = g0; c < min(g0 + ${16 / (subs / 4)}u, n); c++) {
        let x0 = raw4(t0 + c, 0u, sl, alen, ch);
        let y = silu16(w0 * xm3 + w1 * xm2 + w2 * xm1 + w3 * x0);
        xm3 = xm2; xm2 = xm1; xm1 = x0;
        let inv = inverseSqrt(subgroupAdd(dot(y, y)) + 1e-6);
        let at = c * 32u + (lane % ${J}u) * ${lanes}u + lane / ${J}u;
        if (isK) { ks[at] = y * inv; } else { qs[at] = y * (inv * inverseSqrt(128.0)); }
      }
    } else {
      let vt = li - ${threads / 2}u; let chn = vt % 128u; let v0 = (vt / 128u) * ${16 / (threads / 2 / 128)}u;
      if (vt < n) {
        let zr = phys(t0 + vt, sl, alen) * p.stride + p.qkv + QK;
        let b = f32(Z[zr + h]); let a = f32(Z[zr + p.H + h]) + f32(dtb[h]);
        let sp = select(log(1.0 + exp(a)), a, a > 20.0);
        gs[vt] = vec4<f32>(exp(-exp(f32(Alog[h])) * sp), 1.0 / (1.0 + exp(-b)), 0.0, 0.0);
      }
      let ch = 2u * QK + h * 128u + chn;
      let wv = vec4<f32>(f32(W[ch * 4u]), f32(W[ch * 4u + 1u]), f32(W[ch * 4u + 2u]), f32(W[ch * 4u + 3u]));
      var xm3 = raw1(t0 + v0, 3u, sl, alen, ch); var xm2 = raw1(t0 + v0, 2u, sl, alen, ch); var xm1 = raw1(t0 + v0, 1u, sl, alen, ch);
      for (var c = v0; c < min(v0 + ${16 / (threads / 2 / 128)}u, n); c++) {
        let x0 = raw1(t0 + c, 0u, sl, alen, ch);
        let y = dot(wv, vec4<f32>(xm3, xm2, xm1, x0));
        vs[c * 128u + chn] = f32(f16(y / (1.0 + exp(-y))));
        xm3 = xm2; xm2 = xm1; xm1 = x0;
      }
    }
    workgroupBarrier();
    // 2. the recurrence
    for (var c = 0u; c < n; c++) {
      let dec = gs[c].x;
${loads}
${form === "mlx" ? `      var kv = 0.0;
${decay}
      ${shuffles("kv")}
      let delta = (vs[c * 128u + row] - kv) * gs[c].y;
      var o = 0.0;
${upd}
      ${shuffles("o")}` : `      var r = 0.0; var pq = 0.0; var kq = 0.0;
${dots}
      ${shuffles("r")} ${shuffles("pq")} ${shuffles("kq")}
      let delta = (vs[c * 128u + row] - dec * r) * gs[c].y;
      let o = dec * pq + kq * delta;
${step}`}
      if (part == 0u) { vs[c * 128u + row] = o; }
    }
    workgroupBarrier();
    // 3. gated RMSNorm
    for (var c = sub; c < n; c += ${subs}u) {
      if (t0 + c < alen && !writeA) { continue; } // a shared trunk is written by one of its branches only
      let t = phys(t0 + c, sl, alen);
      let x = vec4<f32>(vs[c * 128u + lane * 4u], vs[c * 128u + lane * 4u + 1u], vs[c * 128u + lane * 4u + 2u], vs[c * 128u + lane * 4u + 3u]);
      let inv = inverseSqrt(subgroupAdd(dot(x, x)) / 128.0 + p.eps);
      for (var d = 0u; d < 4u; d++) {
        let z = f32(Z[t * p.stride + p.qkv + h * 128u + lane * 4u + d]);
        Y[t * QK + h * 128u + lane * 4u + d] = f16(x[d] * inv * f32(gw[lane * 4u + d]) * z / (1.0 + exp(-z)));
      }
    }
    workgroupBarrier();
  }
}`;
}

/** Per (token, head) of q (8) and k (2) in the fused [q | k | v | gate] rows: RMSNorm over 256 with the folded
 *  weight, then RoPE (half-split) on the first 64 dims at the token's position; in place. */
export const QK_NORM_ROPE_256 = /* wgsl */ `${HEAD}
struct P { T: u32, heads: u32, kv: u32, stride: u32, eps: f32, theta: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> Z: array<f16>;
@group(0) @binding(2) var<storage, read> qn: array<f16>;
@group(0) @binding(3) var<storage, read> kn: array<f16>;
@group(0) @binding(4) var<storage, read> tmeta: array<vec4<i32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  ${FLAT(64)} let per = p.heads + p.kv; if (i >= p.T * per) { return; }
  let t = i / per; let h = i % per; let isQ = h < p.heads;
  let base = t * p.stride + h * 256u;
  var ss = 0.0;
  for (var d = 0u; d < 256u; d++) { let v = f32(Z[base + d]); ss += v * v; }
  let inv = inverseSqrt(ss / 256.0 + p.eps);
  let ps = f32(tmeta[t].w);
  for (var d = 0u; d < 32u; d++) {
    let a = f32(Z[base + d]) * inv * f32(select(kn[d], qn[d], isQ));
    let b = f32(Z[base + d + 32u]) * inv * f32(select(kn[d + 32u], qn[d + 32u], isQ));
    let ang = ps * pow(p.theta, -f32(2u * d) / 64.0);
    let c = cos(ang); let s = sin(ang);
    Z[base + d] = f16(a * c - b * s);
    Z[base + d + 32u] = f16(b * c + a * s);
  }
  for (var d = 64u; d < 256u; d++) { Z[base + d] = f16(f32(Z[base + d]) * inv * f32(select(kn[d], qn[d], isQ))); }
}`;

/** Attention with 256-dim heads over each token's two key ranges (its sequence, or a shared trunk and its own
 *  branch; bidirectional within them), GQA, output gated by
 *  sigmoid(gate): `qt` queries × `g` query heads of one key/value head per workgroup (so each key/value tile
 *  is loaded once for the g heads), 16 lanes per (query, head) (16 dims each), keys and values through shared
 *  memory 16 at a time. Rows of Z: [q heads | k heads | v heads | gate heads]. Dispatch (ceil(T / qt), heads / g). */
export function attention256(qt = 8, g = 1): string {
  const threads = qt * g * 16;
  return /* wgsl */ `${HEAD}
enable subgroups;
diagnostic(off, subgroup_uniformity);
struct P { T: u32, heads: u32, kv: u32, stride: u32, scale: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Z: array<f16>;
@group(0) @binding(2) var<storage, read> rng: array<vec4<i32>>;    // per token: keys [x, y) and [z, w)
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;      // [T, heads·256]
const QT: u32 = ${qt}u;
const KT: u32 = 16u;
var<workgroup> ks: array<f16, 4096>;
var<workgroup> vs: array<f16, 4096>;
@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let h = wg.y * ${g}u + (li / 16u) % ${g}u; let t0 = wg.x * QT;
  let part = li % 16u; let t = min(t0 + li / ${16 * g}u, p.T - 1u);
  let kvh = h / (p.heads / p.kv);
  let koff = p.heads * 256u + kvh * 256u; let voff = (p.heads + p.kv) * 256u + kvh * 256u;
  var q: array<f32, 16>;
  for (var d = 0u; d < 16u; d++) { q[d] = f32(Z[t * p.stride + h * 256u + part * 16u + d]) * p.scale; }
  let mt = rng[t];
  let last = rng[min(t0 + QT - 1u, p.T - 1u)];
  let kmin = u32(rng[t0].x); let kmax = u32(max(last.y, last.w));
  var m = -1e30; var l = 0.0; var acc: array<f32, 16>;
  for (var kc = kmin; kc < kmax; kc += KT) {
    for (var e = li; e < KT * 256u; e += ${threads}u) {
      let row = min(kc + e / 256u, p.T - 1u) * p.stride;
      ks[e] = Z[row + koff + e % 256u]; vs[e] = Z[row + voff + e % 256u];
    }
    workgroupBarrier();
    let n = min(KT, kmax - kc);
    for (var j = 0u; j < n; j++) {
      var s = 0.0;
      for (var d = 0u; d < 16u; d++) { s += q[d] * f32(ks[j * 256u + part * 16u + d]); }
      s += subgroupShuffleXor(s, 1u); s += subgroupShuffleXor(s, 2u); s += subgroupShuffleXor(s, 4u); s += subgroupShuffleXor(s, 8u);
      let k = i32(kc + j);
      if ((k >= mt.x && k < mt.y) || (k >= mt.z && k < mt.w)) {
        let mn = max(m, s); let corr = exp(m - mn); let ex = exp(s - mn);
        l = l * corr + ex;
        for (var d = 0u; d < 16u; d++) { acc[d] = acc[d] * corr + ex * f32(vs[j * 256u + part * 16u + d]); }
        m = mn;
      }
    }
    workgroupBarrier();
  }
  if (t0 + li / ${16 * g}u < p.T) {
    let goff = (p.heads + 2u * p.kv) * 256u + h * 256u;
    for (var d = 0u; d < 16u; d++) {
      let gate = f32(Z[t * p.stride + goff + part * 16u + d]);
      Y[t * p.heads * 256u + h * 256u + part * 16u + d] = f16(acc[d] / l / (1.0 + exp(-gate)));
    }
  }
}`;
}

/** out[k] = X[row_k] · R[code_k] in f32 (the code readout at each sequence's last token). One workgroup per pair. */
export const READOUT = /* wgsl */ `${HEAD}
struct P { P: u32, H: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> R: array<f32>;
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;  // (token row, code)
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let k = wg.x + wg.y * nw.x; if (k >= p.P) { return; }
  let pr = pairs[k];
  var s = 0.0;
  for (var d = l.x; d < p.H; d += 256u) { s += f32(X[pr.x * p.H + d]) * R[pr.y * p.H + d]; }
  part[l.x] = s; workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (l.x < st) { part[l.x] += part[l.x + st]; } workgroupBarrier(); }
  if (l.x == 0u) { out[k] = part[0]; }
}`;
