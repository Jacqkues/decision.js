// Files of a Hugging Face repo for the tests, cached in .cache/hf.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

export async function hfJson(repo: string, file: string) {
  const path = join(".cache/hf", repo, file);
  if (!existsSync(path)) {
    const res = await fetch(`https://huggingface.co/${repo}/resolve/main/${file}`);
    if (!res.ok) throw new Error(`${repo}/${file}: HTTP ${res.status}`);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, await res.text());
  }
  return JSON.parse(readFileSync(path, "utf8"));
}
