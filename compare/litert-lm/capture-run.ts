// npx tsx compare/litert-lm/capture-run.ts [len]: runs capture.html in Chrome and writes wgsl/NNN.wgsl and trace.txt.
import { chromium } from "playwright";
import { createServer } from "vite";
import { mkdirSync, writeFileSync } from "node:fs";

const len = process.argv[2] ?? "128";
const server = await createServer({ server: { port: 0 }, logLevel: "error" });
await server.listen();
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-webgpu-developer-features", "--enable-gpu", "--use-angle=metal", "--ignore-gpu-blocklist"] });
try {
  const p = await browser.newPage();
  p.on("console", (m) => m.type() === "error" && console.log("[page]", m.text().slice(0, 300)));
  await p.goto(`${server.resolvedUrls!.local[0]}compare/litert-lm/capture.html?len=${len}`);
  await p.waitForFunction(() => (window as any).__r?.done || (window as any).__r?.error, null, { timeout: 600_000 });
  console.log(await p.locator("#out").innerText());
  const r = await p.evaluate(() => (window as any).__r);
  const dir = new URL(`./wgsl-${len}/`, import.meta.url).pathname;
  mkdirSync(dir, { recursive: true });
  r.shaders.forEach((s: { label: string; code: string }, i: number) => writeFileSync(`${dir}${String(i).padStart(3, "0")}.wgsl`, `// ${s.label}\n${s.code}`));
  writeFileSync(`${dir}trace.txt`, r.trace.join("\n"));
} finally {
  await browser.close();
  await server.close();
}
