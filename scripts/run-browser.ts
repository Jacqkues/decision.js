// Serves the repo with Vite and runs a test page in Chrome on the real GPU (WebGPU), printing its output.
//   npx tsx scripts/run-browser.ts parity "model=d1-omni-600M-q8"
import { chromium } from "playwright";
import { createServer } from "vite";

const [page = "parity", query = ""] = process.argv.slice(2);
const server = await createServer({ server: { port: 0 }, logLevel: "error" });
await server.listen();
const url = server.resolvedUrls!.local[0]!;
const flags = process.env.SAFE ? ["--enable-gpu", "--use-angle=metal"] : ["--enable-unsafe-webgpu", "--enable-webgpu-developer-features", "--enable-gpu", "--use-angle=metal", "--ignore-gpu-blocklist"];
const browser = await chromium.launch({ channel: "chrome", headless: true, args: flags });
try {
  const p = await browser.newPage();
  p.on("console", (m) => m.type() === "error" && console.log("[page]", m.text().slice(0, 400)));
  await p.goto(`${url}${page.includes("/") ? page.replace(/^(\.\.\/)+/, "") : `test/browser/${page}`}.html?${query}`);
  await p.waitForFunction(() => (window as any).__r?.done || (window as any).__r?.error, null, { timeout: Number(process.env.TIMEOUT ?? 1_800_000) });
  console.log(await p.locator("#out").innerText());
  const r = await p.evaluate(() => (window as any).__r);
  process.exitCode = r.error ? 1 : 0;
} finally {
  await browser.close();
  await server.close();
}
