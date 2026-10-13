// The d3-lite demo: ?model=<URL of exported weights> (default: /models/d3-lite-q8/, from scripts/export-d3.py).
import { D1, type Answer, type Question } from "../src";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const state = $<HTMLTextAreaElement>("state"), questions = $<HTMLTextAreaElement>("questions");
const run = $<HTMLButtonElement>("run"), bench = $<HTMLButtonElement>("bench"), share = $<HTMLInputElement>("share");
const status = $("status"), progress = $<HTMLProgressElement>("progress"), answers = $("answers"), speed = $("speed");

const presets: Record<string, { state: unknown; questions: Record<string, Question> }> = {
  "Support ticket": {
    state: "Hi, I ordered a laptop three weeks ago and it still hasn't arrived. Tracking says it's stuck in Lyon. I've emailed twice with no answer. If I don't get it by Friday I'm cancelling my subscription and going elsewhere.",
    questions: {
      topic: { type: "choice", instructions: "What is the main topic of the ticket?", criteria: { billing: "billing, payment, refund", delivery: "delivery, parcel", bug: "technical bug", account: "account, login, access", other: null } },
      churn: { type: "noul", instructions: "Is the customer at risk of leaving?" },
      urgency: { type: "score", instructions: "How urgent is the ticket?", criteria: ["low", "medium", "high"] },
    },
  },
  "Tool routing": {
    state: "Can you book me a table for two at an Italian place near the office tomorrow around 8pm, and remind me an hour before?",
    questions: {
      tool: { type: "choice", instructions: "Which tool should handle the main request?", criteria: { search_web: "search the web", book_restaurant: "book a restaurant table", send_email: "send an email", get_weather: "get the weather forecast", set_reminder: "set a reminder" } },
      several_actions: { type: "noul", instructions: "Does the user ask for more than one action?" },
      complexity: { type: "score", instructions: "How complex is the request?", criteria: ["simple", "moderate", "complex"] },
    },
  },
  "Moderation (JSON)": {
    state: { user: "night_owl_42", thread: "Best budget headphones?", comment: "Honestly anyone still buying these is an idiot. Go back to your cave.", reports: 3 },
    questions: {
      toxic: { type: "noul", instructions: "Is the comment insulting or toxic?", criteria: { true: "insulting, harassing or hateful", false: "civil, even if critical" } },
      action: { type: "choice", instructions: "What should the moderator do?", criteria: { keep: "leave it as is", warn: "warn the user", hide: "hide the comment", ban: "ban the user" } },
    },
  },
};

const show = (name: string) => {
  const p = presets[name]!;
  state.value = typeof p.state === "string" ? p.state : JSON.stringify(p.state, null, 2);
  questions.value = JSON.stringify(p.questions, null, 2);
  document.querySelectorAll("#presets button").forEach((b) => b.classList.toggle("on", b.textContent === name));
};
for (const name of Object.keys(presets)) {
  const b = document.createElement("button");
  b.textContent = name;
  b.onclick = () => show(name);
  $("presets").append(b);
}
show("Support ticket");

const model = new URLSearchParams(location.search).get("model") ?? "/models/d3-lite-q8/";
const d3 = await D1.load(model, {
  onProgress: (a, b) => ((progress.value = a / b), (status.textContent = `loading d3-lite… ${(a / 1e6).toFixed(0)} / ${(b / 1e6).toFixed(0)} MB`)),
}).catch((e) => {
  status.textContent = navigator.gpu ? `could not load ${model}: ${e.message}` : "this browser has no WebGPU (try a recent Chrome or Edge)";
  throw e;
});
if (d3.kind !== "d3") status.textContent = `note: ${model} holds ${d3.kind}, not d3-lite`;
progress.hidden = true;
const ready = `${d3.kind} ready · ${d3.engine.matrixUnits ? "matrix units" : "portable kernels"}`;
status.textContent = ready;
run.disabled = bench.disabled = false;

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
const bars = (probs: Record<string, number>, best: string) =>
  Object.entries(probs).map(([k, p]) => `<div class="row ${k === best ? "best" : ""}"><span title="${esc(k)}">${esc(k)}</span><div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div></div><span class="num">${(100 * p).toFixed(1)}%</span></div>`).join("");

function render(name: string, a: Answer) {
  if (a.type === "noul") return `<div class="card"><h3>${esc(name)} · <b>${a.noul >= 0.5 ? "yes" : "no"}</b></h3>${bars({ yes: a.noul, no: 1 - a.noul }, a.noul >= 0.5 ? "yes" : "no")}</div>`;
  if (a.type === "choice") return `<div class="card"><h3>${esc(name)} · <b>${esc(a.choice)}</b></h3>${bars(a.probabilities, a.choice)}</div>`;
  const best = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])[0]![0];
  const label = (k: string) => `${k} · ${a.legend[k]}`;
  return `<div class="card"><h3>${esc(name)} · <b>${esc(a.legend[best]!)}</b> (expected level ${a.score.toFixed(2)})</h3>${bars(Object.fromEntries(Object.entries(a.probabilities).map(([k, p]) => [label(k), p])), label(best))}</div>`;
}

/** The state (JSON if it parses, else text) and the questions, or null after saying what is wrong. */
function inputs(): [unknown, Record<string, Question>] | null {
  let qs: Record<string, Question>, s: unknown = state.value;
  try { qs = JSON.parse(questions.value); } catch { status.textContent = "the questions are not valid JSON"; return null; }
  try { s = JSON.parse(state.value); } catch { /* plain text */ }
  return [s, qs];
}

async function busy(f: () => Promise<void>) {
  run.disabled = bench.disabled = true;
  d3.shareState = share.checked;
  try { await f(); } catch (e) { status.textContent = (e as Error).message; }
  run.disabled = bench.disabled = false;
}

run.onclick = () => busy(async () => {
  const io = inputs(); if (!io) return;
  const t = performance.now();
  const r = await d3.systemOne(io[0], io[1]);
  answers.innerHTML = Object.entries(r.answers).map(([n, a]) => render(n, a as Answer)).join("");
  status.textContent = `${Object.keys(io[1]).length} decisions in ${(performance.now() - t).toFixed(0)} ms · ${r.usage.input_tokens} tokens read${d3.shareState ? " · state read once" : ""}`;
});

bench.onclick = () => busy(async () => {
  const io = inputs(); if (!io) return;
  const n = 100, reqs = Array.from({ length: n }, () => io);
  status.textContent = "warming up…";
  await d3.systemOneBatch(reqs.slice(0, 4));
  status.textContent = `running ${n} requests…`;
  const t = performance.now();
  const res = await d3.systemOneBatch(reqs, { onProgress: (done) => (status.textContent = `running ${n} requests… ${done}/${n}`) });
  const ms = (performance.now() - t) / n, q = Object.keys(io[1]).length, tokens = res.reduce((a, r) => a + r.usage.input_tokens, 0);
  speed.innerHTML = `<b>${ms.toFixed(1)} ms</b> per request (${q} questions) · <b>${(1000 / ms).toFixed(1)}</b> requests/s · <b>${((tokens / n / ms) * 1000).toFixed(0)}</b> tokens/s${d3.shareState ? " · state read once" : ""}`;
  status.textContent = ready;
});
