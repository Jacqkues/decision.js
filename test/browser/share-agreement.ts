// d3-lite with shareState against d3-lite without it, on the 300 benchmark tickets (3 questions each):
// how often the answer is the same, and the largest probability gap. ?model=d3-lite-q8
import { D1 } from "../../src";
import { questions, tickets } from "./tickets";
const out = document.getElementById("out")!;
const R: Record<string, unknown> = ((window as any).__r = {});
const argmax = (p: number[]) => p.indexOf(Math.max(...p));
try {
  const d1 = await D1.load(`/models/${new URLSearchParams(location.search).get("model") ?? "d3-lite-q8"}/`);
  const reqs = tickets(300).map((t) => [t, Object.values(questions)] as [string, (typeof questions)[string][]]);
  const exact = await d1.probabilitiesBatch(reqs);
  d1.shareState = true;
  const shared = await d1.probabilitiesBatch(reqs);
  const names = Object.keys(questions);
  names.forEach((name, j) => {
    let same = 0, worst = 0;
    exact.forEach((p, i) => { same += +(argmax(p[j]!) === argmax(shared[i]![j]!)); worst = Math.max(worst, ...p[j]!.map((v, k) => Math.abs(v - shared[i]![j]![k]!))); });
    out.textContent += `${name.padEnd(12)} same answer ${same}/300 · largest probability gap ${worst.toFixed(3)}\n`;
  });
  R.done = true;
} catch (e) { out.textContent += String((e as Error).stack); R.error = String(e); }
