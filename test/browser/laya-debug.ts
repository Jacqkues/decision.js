// Debugging: Laya's hidden states, layer by layer, for one fixture sequence (?state=6&question=2).
import { LayaEngine } from "../../src/laya-engine";
import lx from "../fixtures/laya-prompts.json";
const out = document.getElementById("out")!;
const R: Record<string, unknown> = ((window as any).__r = {});
const q = new URLSearchParams(location.search);
try {
  const c = (lx.cases as any[]).find((c) => c.state === Number(q.get("state") ?? 6) && c.question === Number(q.get("question") ?? 2) && c.ids)!;
  const e = await LayaEngine.load("/models/laya-q8/", { directWeights: q.get("direct") === "0" ? false : undefined });
  const layers = [27, -1, -2];
  const { debug } = await e.logits([{ ids: c.ids, markers: c.markers, type: "noul" }], layers);
  const H = 1024;
  for (const l of layers) {
    const h = debug[l]!;
    let mx = 0; for (let i = 0; i < c.ids.length * H; i++) mx = Math.max(mx, Math.abs(h[i]!));
    const at = (t: number) => Array.from(h.subarray(t * H, t * H + 4)).map((v) => v.toFixed(3)).join(" ");
    out.textContent += `layer ${l}: max |h| ${mx.toFixed(1)} · token 5: ${at(5)} · markers: ${c.markers.map((m: number) => at(m)).join(" | ")}\n`;
  }
  R.done = true;
} catch (e) { out.textContent += String((e as Error).stack); R.error = String(e); }
