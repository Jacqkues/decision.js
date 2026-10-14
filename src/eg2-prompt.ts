// The texts of the EmbeddingGemma 2 typed-decision model (modeling_gemmadecision.py: render_state,
// parse_question, render_question, render_option) in TypeScript. The model embeds each text on its own:
//   state:    "workflow: W\nkey: value\nparent · child: value…" (a dict, flattened), else "workflow: W | state: S"
//   question: "<type> question: instructions"
//   option:   "question: instructions | answer: <option>" (noul: "no", "yes"; score: "level i of n: description")
// each prefixed by the task prompt ("task: classification | query: ") and tokenized with <bos> … <eos>.
import { pyStr } from "./py.js";
import type { Question } from "./questions.js";
import type { TextTokenizer } from "./tokenizer.js";

/** The state's text. A string that holds a JSON object or array is read as that value, as the model does
 *  (it fails on an empty string, or one that only starts like JSON: both are rendered as plain text here). */
export function renderState(state: unknown, workflow = ""): string {
  if (typeof state === "string" && /^\s*[{[]/.test(state)) {
    try { state = JSON.parse(state); } catch { /* plain text */ }
  }
  if (state === null || typeof state !== "object" || Array.isArray(state)) return `workflow: ${workflow} | state: ${pyStr(state)}`;
  const lines: string[] = [];
  const walk = (prefix: string, v: unknown) => {
    if (v !== null && typeof v === "object" && !Array.isArray(v)) for (const [k, x] of Object.entries(v)) walk(prefix ? `${prefix} · ${k}` : k, x);
    else lines.push(`${prefix.replaceAll("_", " ")}: ${pyStr(v)}`);
  };
  walk("", state);
  return `workflow: ${workflow}\n` + lines.join("\n");
}

/** The question's text and its option texts, in option order ([no, yes] for a noul). `name`: the text used when
 *  the instructions are empty. */
export function renderQuestion(q: Question, name = ""): { question: string; options: string[] } {
  const instr = q.instructions ? String(q.instructions) : name;
  const answer = (a: string) => `question: ${instr} | answer: ${a}`;
  let options: string[];
  if (q.type === "noul") options = [answer("no"), answer("yes")];
  else if (q.type === "score") options = q.criteria.map((c, i) => answer(`level ${i} of ${q.criteria.length - 1}: ${pyStr(c)}`));
  else options = Object.entries(q.criteria).map(([k, v]) => answer(pyStr(v) === k ? k : `${k}: ${pyStr(v)}`));
  return { question: `${q.type} question: ${instr}`, options };
}

/** Token ids of one text as the model embeds it: <bos> prompt + text <eos>, at most `maxLen` tokens. */
export function tokenize(tok: TextTokenizer, text: string, prompt: string, maxLen = 8192): number[] {
  const ids = tok.encode(prompt + text);
  return [tok.id(tok.special("bos_token")), ...ids.slice(0, maxLen - 2), tok.id(tok.special("eos_token"))];
}
