// Debugging: d3-lite's hidden state after every layer vs PyTorch fp32 (test/fixtures/d3-prompts.json's debug
// dump), as cosine similarity at a few tokens. ?model=d3-lite-q8 &direct=0 &causal=1
// The dump comes from a plain forward of the language model, which d3's noncausal attention hook does not
// reach: compare with causal=1. Without it, the final logits are the ones to check against the reference.
import { D3Engine } from "../../src/d3-engine";
import dx from "../fixtures/d3-prompts.json";
const out = document.getElementById("out")!;
const R: Record<string, unknown> = ((window as any).__r = {});
const q = new URLSearchParams(location.search);
const cos = (a: ArrayLike<number>, b: number[]) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < b.length; i++) (d += a[i]! * b[i]!), (na += a[i]! ** 2), (nb += b[i]! ** 2); return d / Math.sqrt(na * nb); };
try {
  const dbg = (dx as any).debug, c = (dx as any).cases.find((c: any) => c.state === dbg.state && c.question === dbg.question);
  if (q.get("causal")) (globalThis as any).__d3Causal = true;
  const e = await D3Engine.load(`/models/${q.get("model") ?? "d3-lite-q8"}/`, { directWeights: q.get("direct") === "0" ? false : undefined });
  const layers = Array.from({ length: 24 }, (_, i) => i);
  const { logits, debug } = await e.logits([{ ids: c.ids, options: c.options }], [...layers, -1]);
  const H = 1024;
  const line = (name: string, buf: Float32Array, ref: number[][]) =>
    (out.textContent += `${name.padEnd(9)} ${dbg.tokens.map((t: number, j: number) => cos(buf.subarray(t * H, t * H + H), ref[j]!).toFixed(4)).join("  ")}\n`);
  for (const l of layers) line(`layer ${l}`, debug[l]!, dbg.hidden[l]);
  line("final", debug[-1]!, dbg.final);
  out.textContent += `logits ${logits[0]!.map((v) => v.toFixed(3)).join(" ")} · reference probabilities ${c.probabilities.map((v: number) => v.toFixed(3)).join(" ")}\n`;
  R.done = true;
} catch (e) { out.textContent += String((e as Error).stack); R.error = String(e); }
