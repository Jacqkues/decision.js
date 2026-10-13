// Laya's prompt (laya/common.py build_head + build_sequence, Agent._encode_state) in TypeScript. One
// sequence per question:
//   [CLS] <type> question: instructions [SEP] [MASK] opt_0 [MASK] opt_1 … [SEP] state [SEP]
// with the model scoring the hidden state at every [MASK]. The question part gets head_max_len tokens
// (options first, 48 tokens each at most), the state what is left of max_len; a list state (a
// conversation) loses its oldest tokens, anything else its last ones.
import { pyDumps } from "./py.js";
import type { Question } from "./questions.js";
import type { TextTokenizer } from "./tokenizer.js";

const criterion = (v: unknown) => (typeof v === "string" ? v : pyDumps(v));
const serialize = (state: unknown) => (typeof state === "string" ? state : pyDumps(state));

/** Option texts in the model's order; a noul is read as [false, true]. */
export function renderOptions(q: Question): string[] {
  if (q.type === "choice") return Object.entries(q.criteria).map(([k, v]) => (v == null || v === "" ? k : `${k}: ${criterion(v)}`));
  if (q.type === "score") return q.criteria.map((c, i) => `level ${i}: ${criterion(c)}`);
  const crit = (q.criteria ?? {}) as Record<string, unknown>;
  for (const k of Object.keys(crit)) if (k !== "true" && k !== "false") throw new Error("decision.js: a Laya noul question takes criteria keyed only 'true'/'false'");
  const no = crit.false, yes = crit.true;
  return [
    "false: " + (no != null && no !== "" ? criterion(no) : "no, the statement does not hold"),
    "true: " + (yes != null && yes !== "" ? criterion(yes) : "yes, the statement holds"),
  ];
}

export function encode(tok: TextTokenizer, state: unknown, q: Question, maxLen = 512, headMaxLen = 192): { ids: number[]; markers: number[] } {
  if (state === null || state === undefined) throw new Error("decision.js: Laya needs a state (a string, an object or a list)");
  const mask = tok.special("mask_token"), maskId = tok.id(mask), cls = tok.id(tok.special("cls_token")), sep = tok.id(tok.special("sep_token"));
  const clean = (s: string) => s.replaceAll(mask, " ");
  const opts = renderOptions(q);
  let headIds = tok.encode(`${q.type} question: ${clean(String(q.instructions))}`);
  let optIds = opts.map((o) => [maskId, ...tok.encode(" " + clean(o)).slice(0, 48)]);
  let budget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
  if (budget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    budget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
  }
  headIds = headIds.slice(0, Math.max(8, budget));
  let ids = [cls, ...headIds, sep];
  let markers: number[] = [];
  for (const o of optIds) (markers.push(ids.length), ids.push(...o));
  ids.push(sep);
  const room = Math.max(0, maxLen - ids.length - 1);
  const stateIds = tok.encode(clean(serialize(state)));
  const st = Array.isArray(state) ? stateIds.slice(Math.max(0, stateIds.length - room)) : stateIds.slice(0, room);
  ids = [...ids, ...st, sep].slice(0, maxLen);
  markers = markers.filter((m) => m < maxLen);
  if (markers.length !== opts.length) throw new Error(`decision.js: only ${markers.length} of the ${opts.length} option markers fit in ${maxLen} tokens`);
  return { ids, markers };
}
