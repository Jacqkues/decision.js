// d1-3B's prompt.py in TypeScript (its defaults: no system turn, `json_only` states, `desc` options).
// A decision is one causal pass that stops at the answer slot: the state is a prefix shared by every
// question (the trunk), each question a suffix (a branch), and the answer a softmax over its options' tokens.
import { pyDumps, pyStr, pyTruthy } from "./py.js";
import type { Question } from "./questions.js";
import type { TextTokenizer } from "./tokenizer.js";

const IM_START = "<|im_start|>", IM_END = "<|im_end|>";

/** Native letters when the labels already are letters, else A..Z, else 00.. */
function optionCodes(labels: string[]): string[] {
  const labs = labels.map((x) => String(x).trim());
  if (labs.length && labs.every((k) => [...k].length === 1 && /^\p{L}$/u.test(k))) return labs;
  if (labs.length <= 26) return labs.map((_, i) => String.fromCharCode(65 + i));
  return labs.map((_, i) => String(i).padStart(2, "0"));
}

const AZ = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
const FALLBACK_POOL = [
  ...AZ,
  ...Array.from({ length: 100 }, (_, i) => String(i).padStart(2, "0")),
  ...AZ.map((c) => c.toLowerCase()),
  ...Array.from({ length: 200 }, (_, i) => `#${i}`),
  ...AZ.flatMap((a) => AZ.map((b) => a + b)),
];

export class D13BPrompt {
  private aliasCache = new Map<string, [string, number][]>();
  private idsCache = new Map<string, number[]>();

  constructor(readonly tok: TextTokenizer) {}

  /** Every label gets a distinct single-token code: [(code, token id)]. */
  aliases(labels: string[]): [string, number][] {
    const codes = optionCodes(labels);
    const key = codes.join("\u0000");
    let out = this.aliasCache.get(key);
    if (out) return out;
    out = [];
    const used = new Set<number>();
    const take = (raw: string) => {
      const enc = this.tok.encode(raw);
      if (enc.length !== 1 || used.has(enc[0]!)) return false;
      out!.push([raw, enc[0]!]);
      used.add(enc[0]!);
      return true;
    };
    for (const code of codes) {
      if (take(code)) continue;
      if (!FALLBACK_POOL.some(take)) throw new Error(`decision.js: no single-token alias left for ${codes.length} options`);
    }
    this.aliasCache.set(key, out);
    return out;
  }

  /** Single-token ids of `texts`, without repeats. */
  private ids(texts: string[]): number[] {
    const key = texts.join("\u0000");
    let out = this.idsCache.get(key);
    if (out) return out;
    out = [];
    for (const t of texts) {
      const enc = this.tok.encode(t);
      if (enc.length === 1 && !out.includes(enc[0]!)) out.push(enc[0]!);
    }
    this.idsCache.set(key, out);
    return out;
  }

  /** Token ids to score, one group per option ([yes, no] for a noul): each option scores its best form. */
  readoutIds(q: Question): number[][] {
    if (q.type === "noul") {
      const yes = this.ids(["yes", "Yes", "YES"]), no = this.ids(["no", "No", "NO"]);
      if (!yes.length || !no.length) throw new Error("decision.js: tokenizer has no single-token yes/no");
      return [yes, no];
    }
    if (q.type === "score") {
      const groups = q.criteria.map((_, i) => this.ids([String(i)]));
      if (groups.some((g) => !g.length)) throw new Error(`decision.js: a score with ${q.criteria.length} levels needs single-token digits`);
      return groups;
    }
    return this.aliases(Object.keys(q.criteria)).map(([code, tid]) => [tid, ...this.ids([` ${code}`]).filter((i) => i !== tid)]);
  }

  questionBlock(q: Question): string {
    if (q.type === "choice") {
      const labels = Object.keys(q.criteria);
      const codes = this.aliases(labels);
      const lines = labels.map((lab, i) => {
        const desc = q.criteria[lab];
        return `${codes[i]![0]} ${pyTruthy(desc) ? pyStr(desc) : lab.replaceAll("_", " ")}`;
      }).join("\n");
      return `${q.instructions}\n\nOptions:\n${lines}\n\nReply with the option code only.`;
    }
    if (q.type === "noul") {
      const c = q.criteria as Record<string, unknown> | null | undefined;
      const extra = c && pyTruthy(c) ? `\nYes: ${pyStr(c.true)}\nNo: ${pyStr(c.false)}` : "";
      return `${q.instructions}${extra}\n\nReply with yes or no only.`;
    }
    const legend = q.criteria.map((name, i) => `${i} ${pyStr(name)}`).join("\n");
    return `${q.instructions}\n\n${legend}\n\nReply with a single digit 0-${q.criteria.length - 1} only.`;
  }

  /** Everything before the question, shared by all questions on one state (no state: null). */
  prefixText(state: unknown): string {
    const block = typeof state === "string" ? `${state}\n\n` : pyDumps(state, 2) + "\n\n";
    const body = state === null || state === undefined ? "" : `${block}\nQUESTION:\n`;
    return `${this.tok.bos}${IM_START}user\n${body}`;
  }

  /** The question and the assistant header, up to the answer slot. */
  suffixText(q: Question): string {
    return `${this.questionBlock(q)}${IM_END}\n${IM_START}assistant\n`;
  }

  trunk(state: unknown) {
    return this.tok.encode(this.prefixText(state));
  }

  branch(q: Question) {
    return this.tok.encode(this.suffixText(q));
  }
}
