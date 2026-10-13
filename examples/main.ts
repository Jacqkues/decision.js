// The d1.js demo: ?model=<URL of exported weights> (default: /models/d1-omni-600M-q8/).
import { D1, type Answer, type Question } from "../src";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const state = $<HTMLTextAreaElement>("state"), questions = $<HTMLTextAreaElement>("questions");
const run = $<HTMLButtonElement>("run"), status = $("status"), answers = $("answers");

state.value = "Hi, I ordered a laptop three weeks ago and it still hasn't arrived. Tracking says it's stuck in Lyon. I've emailed twice with no answer. If I don't get it by Friday I'm cancelling my subscription and going elsewhere.";
questions.value = JSON.stringify({
  topic: { type: "choice", instructions: "What is the main topic of the ticket?", criteria: { billing: "billing, payment", delivery: "delivery, parcel", bug: "technical bug", account: "account, access", other: null } },
  churn: { type: "noul", instructions: "Is the customer at risk of leaving?" },
  urgency: { type: "score", instructions: "How urgent is the ticket?", criteria: ["low", "medium", "high"] },
}, null, 2);

const model = new URLSearchParams(location.search).get("model") ?? "/models/d1-omni-600M-q8/";
const d1 = await D1.load(model, { onProgress: (a, b) => (status.textContent = `loading the model… ${Math.round((100 * a) / b)}%`) }).catch((e) => {
  status.textContent = `could not load ${model}: ${e.message}`;
  throw e;
});
status.textContent = `${d1.kind} ready · ${d1.engine.matrixUnits ? "matrix units" : "portable kernels"}`;
run.disabled = false;

const bars = (probs: Record<string, number>, best: string) =>
  Object.entries(probs).map(([k, p]) => `<div class="row ${k === best ? "best" : ""}"><span>${k}</span><div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div></div><span class="num">${(100 * p).toFixed(1)}%</span></div>`).join("");

function render(name: string, a: Answer) {
  if (a.type === "noul") return `<div class="card"><h3>${name} · ${a.noul >= 0.5 ? "yes" : "no"}</h3>${bars({ yes: a.noul, no: 1 - a.noul }, a.noul >= 0.5 ? "yes" : "no")}</div>`;
  if (a.type === "choice") return `<div class="card"><h3>${name} · ${a.choice}</h3>${bars(a.probabilities, a.choice)}</div>`;
  const best = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])[0]![0];
  return `<div class="card"><h3>${name} · level ${a.score.toFixed(2)}</h3>${bars(Object.fromEntries(Object.entries(a.probabilities).map(([k, p]) => [`${k} ${a.legend[k]}`, p])), `${best} ${a.legend[best]}`)}</div>`;
}

run.onclick = async () => {
  let qs: Record<string, Question>, s: unknown = state.value;
  try { qs = JSON.parse(questions.value); } catch { status.textContent = "the questions are not valid JSON"; return; }
  try { s = JSON.parse(state.value); } catch { /* plain text */ }
  run.disabled = true;
  const t = performance.now();
  try {
    const r = await d1.systemOne(s, qs);
    answers.innerHTML = Object.entries(r.answers).map(([n, a]) => render(n, a as Answer)).join("");
    status.textContent = `${Object.keys(qs).length} decisions in ${(performance.now() - t).toFixed(0)} ms · ${r.usage.input_tokens} tokens read`;
  } catch (e) {
    status.textContent = (e as Error).message;
  }
  run.disabled = false;
};
