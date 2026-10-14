// Serves the repo with Vite and runs a test or benchmark page in Chrome on the real GPU (WebGPU), printing its
// output (see bench/README.md):
//   npx tsx scripts/run-browser.ts parity "model=d1-omni-600M-q8"       (or npm run bench -- parity "model=…")
// The page is one of the names below or a path to an .html page without its extension. SAFE=1 runs a stock
// Chrome (no --enable-unsafe-webgpu, so no matrix units).
import { chromium } from "playwright";
import { createServer } from "vite";

const PAGES: Record<string, string> = {
  parity: "test/browser/parity",                    // answers against each model's own PyTorch code
  tickets: "test/browser/bench",                    // ms per support ticket, 3 questions each
  embed: "test/browser/embed",                      // EmbeddingGemma 2: ms per text, per kernel, per request
  "typed-decisions": "bench/typed-decisions/index", // accuracy on LocalLLaMA/typed-decisions (test split)
  "decision-index": "bench/decision-index/index",   // accuracy and agreement on a Decision Index sample
  litert: "compare/litert/bench",                   // LiteRT.js (ML Drift) on d1-omni
  "litert-lm": "compare/litert-lm/capture",         // LiteRT-LM (ML Drift) on EmbeddingGemma 2
};
const [page = "parity", query = ""] = process.argv.slice(2);
const path = PAGES[page] ?? page.replace(/^(\.\.\/)+/, "");
const server = await createServer({ server: { port: 0 }, logLevel: "error" });
await server.listen();
const url = server.resolvedUrls!.local[0]!;
const flags = process.env.SAFE ? ["--enable-gpu", "--use-angle=metal"] : ["--enable-unsafe-webgpu", "--enable-webgpu-developer-features", "--enable-gpu", "--use-angle=metal", "--ignore-gpu-blocklist"];
const browser = await chromium.launch({ channel: "chrome", headless: true, args: flags });
try {
  const p = await browser.newPage();
  p.on("console", (m) => m.type() === "error" && !/^(INFO|WARNING): /.test(m.text()) && console.log("[page]", m.text().slice(0, 400)));
  await p.goto(`${url}${path}.html?${query}`);
  await p.waitForFunction(() => (window as any).__r?.done || (window as any).__r?.error, null, { timeout: Number(process.env.TIMEOUT ?? 1_800_000) });
  console.log(await p.locator("#out").innerText());
  const r = await p.evaluate(() => (window as any).__r);
  process.exitCode = r.error ? 1 : 0;
} finally {
  await browser.close();
  await server.close();
}
