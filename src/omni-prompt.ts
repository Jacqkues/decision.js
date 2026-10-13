// d1-omni-600M's prompt.py in TypeScript: one question over one state as one sequence,
//   <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> <opt> <mask> option_1 </opt> ... <decide>
// and the position of every option's <mask>, where the model scores it. Delimiters come from the
// tokenizer's reserved block; `<|...|>` in caller text is rewritten so a state can never forge one.
import { pyDumps } from "./py.js";
import type { Question } from "./questions.js";
import type { TextTokenizer } from "./tokenizer.js";

const DELIM = { state: "<|reserved_7|>", q: "<|reserved_8|>", opt: "<|reserved_9|>", optEnd: "<|reserved_10|>", decide: "<|reserved_11|>" };
const MARKER = "<|mask|>";

/** `<|name|>` → `<¦name¦>`, so caller text cannot emit a delimiter or marker token. */
export const escape = (text: string) => text.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");

const serialize = (state: unknown) => (typeof state === "string" ? state : pyDumps(state));
export const criterion = (v: unknown) => (typeof v === "string" ? v : pyDumps(v));

/** Option texts in the model's order; a noul is read as [false, true]. */
export function renderOptions(q: Question): string[] {
  if (q.type === "choice") return Object.entries(q.criteria).map(([k, v]) => (v == null || v === "" ? k : `${k}: ${criterion(v)}`));
  if (q.type === "score") return q.criteria.map((c, i) => `level ${i}: ${criterion(c)}`);
  const crit = (q.criteria && Object.keys(q.criteria).length ? q.criteria : {}) as Record<string, unknown>;
  const no = "false" in crit ? crit.false : crit.no, yes = "true" in crit ? crit.true : crit.yes;
  return [
    "false: " + (no != null && no !== "" ? criterion(no) : "no, the statement does not hold"),
    "true: " + (yes != null && yes !== "" ? criterion(yes) : "yes, the statement holds"),
  ];
}

/** Token ids of one question over one state, and the position of each option's marker. The option block
 *  gets max(96, min(24k + 32, maxLen / 2)) tokens, shared out evenly; the state is truncated on the right
 *  to the room that is left. */
export function encode(tok: TextTokenizer, state: unknown, q: Question, maxLen = 16384, perOption = 24): { ids: number[]; markers: number[] } {
  const enc = (s: string) => tok.encode(escape(s));
  const opts = renderOptions(q);
  const budget = Math.max(96, Math.min(opts.length * perOption + 32, Math.floor(maxLen / 2)));
  const per = Math.max(2, Math.floor((budget - 3 * opts.length) / opts.length));
  const question = [tok.id(DELIM.q), ...enc(String(q.instructions))].slice(0, Math.max(16, budget));
  const markers: number[] = [];
  for (const text of opts) {
    markers.push(question.length + 1);
    question.push(tok.id(DELIM.opt), tok.id(MARKER), ...enc(" " + text).slice(0, per), tok.id(DELIM.optEnd));
  }
  question.push(tok.id(DELIM.decide));
  const room = Math.max(0, maxLen - question.length - 2);
  const stateIds = [tok.id(DELIM.state), ...enc(serialize(state ?? "")).slice(0, room)];
  const ids = [tok.id(tok.bos), ...stateIds, ...question].slice(0, maxLen);
  const at = markers.map((m) => m + 1 + stateIds.length);
  if (at[at.length - 1]! >= maxLen) throw new Error("decision.js: the options do not fit in the context");
  return { ids, markers: at };
}

/** The calibration key of a question (prompt.py temperature_key). */
export function temperatureKey(q: Question) {
  const k = q.type === "noul" ? 2 : q.type === "choice" ? Object.keys(q.criteria).length : q.criteria.length;
  return `${q.type}:` + (k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+");
}
