// Kernels of the BERT-style encoders (Laya's ModernBERT): RoPE without qk-norm, GeGLU.

const HEAD = /* wgsl */ `enable f16;`;

/** qkv [T, (heads + 2·kv)·64] → half-split RoPE on every q and k head at each token's position, in place
 *  (theta from the uniform: ModernBERT uses one per layer type). */
export const ROPE = /* wgsl */ `${HEAD}
struct P { T: u32, heads: u32, kv: u32, pad: u32, theta: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> QKV: array<f16>;
@group(0) @binding(2) var<storage, read> pos: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  // a long batch needs more than 65,535 workgroups: the dispatch is then 2-D
  let idx = g.x + g.y * nw.x * 64u; let perTok = p.heads + p.kv; if (idx >= p.T * perTok * 32u) { return; }
  let d = idx % 32u; let th = idx / 32u; let t = th / perTok; let h = th % perTok;
  let base = t * (p.heads + 2u * p.kv) * 64u + h * 64u;
  let ang = f32(pos[t]) * pow(p.theta, -f32(2u * d) / 64.0);
  let c = cos(ang); let s = sin(ang);
  let a = f32(QKV[base + d]); let b = f32(QKV[base + d + 32u]);
  QKV[base + d] = f16(a * c - b * s);
  QKV[base + d + 32u] = f16(b * c + a * s);
}`;

/** h[T,I] = gelu(a[:, :I]) · a[:, I:] (a = Wi output, GELU with erf). */
export const GELU_MUL = /* wgsl */ `${HEAD}
struct P { T: u32, I: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read_write> Hh: array<f16>;
fn erf(x: f32) -> f32 { // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
  return select(-y, y, x >= 0.0);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u; if (i >= p.T * p.I) { return; }
  let t = i / p.I; let c = i % p.I;
  let a = f32(A[t * 2u * p.I + c]); let b = f32(A[t * 2u * p.I + p.I + c]);
  Hh[i] = f16(0.5 * a * (1.0 + erf(a * 0.70710678)) * b);
}`;
