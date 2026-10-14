// Downloads a decision.js export hosted on Hugging Face into models/<name>/, for the tests and benchmarks:
//   npx tsx scripts/fetch-model.ts eg2-q8        (or npm run fetch-model -- eg2-q8)
// The other models are exported from their original checkpoints with scripts/export-*.py (see the README).
import { createWriteStream, existsSync, mkdirSync, statSync } from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const HOSTED: Record<string, string> = {
  "d3-lite-q8": "https://huggingface.co/Jacqkues/d3-lite-decisionjs/resolve/main/",
  "eg2-q8": "https://huggingface.co/Jacqkues/embeddinggemma-2-typed-decision-decisionjs/resolve/main/",
  "eg2-q4": "https://huggingface.co/Jacqkues/embeddinggemma-2-typed-decision-decisionjs/resolve/main/q4/",
};
const FILES = ["manifest.json", "tokenizer.json", "tokenizer_config.json", "reference.json", "weights.bin"];

const names = process.argv.slice(2);
if (!names.length || names.some((n) => !HOSTED[n])) {
  console.error(`usage: npx tsx scripts/fetch-model.ts <${Object.keys(HOSTED).join(" | ")}> …`);
  process.exit(1);
}
for (const name of names) {
  const dir = `models/${name}`;
  mkdirSync(dir, { recursive: true });
  for (const file of FILES) {
    const path = `${dir}/${file}`, url = HOSTED[name] + file;
    const head = await fetch(url, { method: "HEAD" });
    const size = Number(head.headers.get("x-linked-size") ?? head.headers.get("content-length")) || 0; // 0: unknown (a small, compressed file)
    if (existsSync(path) && (!size || statSync(path).size === size)) continue; // already there
    process.stdout.write(`${path} (${(size / 1e6).toFixed(0)} MB)… `);
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
    await pipeline(Readable.fromWeb(res.body as never), createWriteStream(path));
    console.log("done");
  }
}
