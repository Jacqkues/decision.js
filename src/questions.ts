// Questions in the Decision Index schema, and the typed answers d1 returns for them.

/** A yes/no question: P(yes). `criteria` optionally says what yes and no mean ({true, false} or {yes, no}). */
export interface Noul { type: "noul"; instructions: string; criteria?: Record<string, unknown> | null }
/** One option out of several: `criteria` maps each option name to its description (or null). */
export interface Choice { type: "choice"; instructions: string; criteria: Record<string, unknown> }
/** A level on a scale: `criteria` lists 2 to 10 level descriptions, lowest first. */
export interface Score { type: "score"; instructions: string; criteria: unknown[] }
export type Question = Noul | Choice | Score;

export type Answer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> }
  | { type: "score"; score: number; confidence: number; probabilities: Record<string, number>; legend: Record<string, string> };

export interface Response<N extends string = string> { answers: Record<N, Answer>; usage: { input_tokens: number; output_tokens: 0 } }

/** Throws on a malformed question, as prompt.py's Question does. */
export function checkQuestion(q: Question): Question {
  if (!q || typeof q !== "object" || !("type" in q) || !("instructions" in q)) throw new Error("decision.js: a question is an object with `type`, `instructions` and, for choice and score, `criteria`");
  if (!["noul", "choice", "score"].includes(q.type)) throw new Error(`decision.js: question type must be noul, choice or score, got ${JSON.stringify(q.type)}`);
  if (q.type === "choice" && (!q.criteria || typeof q.criteria !== "object" || Array.isArray(q.criteria) || Object.keys(q.criteria).length < 2))
    throw new Error("decision.js: a choice needs criteria {name: description} with at least two options");
  if (q.type === "score" && (!Array.isArray(q.criteria) || q.criteria.length < 2 || q.criteria.length > 10))
    throw new Error("decision.js: a score needs criteria: a list of 2 to 10 level descriptions, lowest first");
  if (q.type === "noul" && q.criteria != null && (typeof q.criteria !== "object" || Array.isArray(q.criteria)))
    throw new Error('decision.js: noul criteria are optional: {"true": "...", "false": "..."} (or "yes", "no")');
  return q;
}

/** A question's typed answer from its probabilities ([yes, no] for a noul, else option order). */
export function answer(q: Question, probs: number[], legend: (c: unknown) => string = String): Answer {
  if (q.type === "noul") return { type: "noul", noul: probs[0]! };
  let best = 0;
  probs.forEach((p, i) => p > probs[best]! && (best = i));
  if (q.type === "choice") {
    const names = Object.keys(q.criteria);
    return { type: "choice", choice: names[best]!, confidence: probs[best]!, probabilities: Object.fromEntries(names.map((n, i) => [n, probs[i]!])) };
  }
  return {
    type: "score", score: probs.reduce((a, p, i) => a + i * p, 0), confidence: probs[best]!,
    probabilities: Object.fromEntries(probs.map((p, i) => [String(i), p])),
    legend: Object.fromEntries(q.criteria.map((c, i) => [String(i), legend(c)])),
  };
}
