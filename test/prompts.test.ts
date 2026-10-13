// The TypeScript prompts against LiquidAI's prompt.py (test/fixtures/prompts.json, from
// scripts/make-prompt-fixtures.py): the same token ids, markers and readout groups, token for token.
import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { D13BPrompt } from "../src/d1-3b-prompt";
import { encode } from "../src/omni-prompt";
import type { Question } from "../src/questions";
import { TextTokenizer } from "../src/tokenizer";
import { hfJson } from "./hf";

const fx = JSON.parse(readFileSync("test/fixtures/prompts.json", "utf8"));
const states: unknown[] = fx.states, questions: Question[] = fx.questions;
const tokenizer = async (repo: string, dir = "") => TextTokenizer.from(await hfJson(repo, `${dir}tokenizer.json`), await hfJson(repo, `${dir}tokenizer_config.json`));

describe("d1-omni prompt", () => {
  let tok: TextTokenizer;
  beforeAll(async () => void (tok = await tokenizer("LiquidAI/d1-omni-600M")), 120_000);
  for (const c of fx.omni) {
    it(`state ${c.state} · question ${c.question} · max_len ${c.max_len}`, () => {
      const run = () => encode(tok, states[c.state], questions[c.question]!, c.max_len);
      if (c.error) expect(run).toThrow();
      else expect(run()).toEqual({ ids: c.ids, markers: c.markers });
    });
  }
});

describe("d1-3B prompt", () => {
  let p: D13BPrompt;
  beforeAll(async () => void (p = new D13BPrompt(await tokenizer("LiquidAI/d1-3B"))), 120_000);
  states.forEach((s, i) => it(`trunk of state ${i}`, () => expect(p.trunk(s)).toEqual(fx.d1_3b.trunks[i])));
  questions.forEach((q, i) => {
    it(`branch of question ${i}`, () => expect(p.branch(q)).toEqual(fx.d1_3b.branches[i]));
    it(`readout groups of question ${i}`, () => expect(p.readoutIds(q)).toEqual(fx.d1_3b.groups[i]));
  });
});

describe("Laya prompt", () => {
  const lx = JSON.parse(readFileSync("test/fixtures/laya-prompts.json", "utf8"));
  let tok: TextTokenizer;
  beforeAll(async () => void (tok = await tokenizer("convaiinnovations/laya", "tokenizer/")), 120_000);
  for (const c of lx.cases) {
    it(`state ${c.state} · question ${c.question}`, async () => {
      const { encode: encodeLaya } = await import("../src/laya-prompt");
      const run = () => encodeLaya(tok, states[c.state], questions[c.question]!, lx.config.max_len, lx.config.head_max_len);
      if (c.error) expect(run).toThrow();
      else expect(run()).toEqual({ ids: c.ids, markers: c.markers });
    });
  }
});

describe("d3-lite prompt", () => {
  const dx = JSON.parse(readFileSync("test/fixtures/d3-prompts.json", "utf8"));
  let tok: TextTokenizer, codes: string[];
  beforeAll(async () => {
    tok = await tokenizer("vllm-sr/d3-lite");
    codes = (await hfJson("vllm-sr/d3-lite", "decision_config.json")).codes;
  }, 300_000);
  for (const c of dx.cases) {
    it(`state ${c.state} · question ${c.question}`, async () => {
      const { encode: encodeD3 } = await import("../src/d3-prompt");
      const run = () => encodeD3(tok, states[c.state], questions[c.question]!, codes);
      if (c.error) expect(run).toThrow();
      else expect(run()).toEqual({ ids: c.ids, options: c.options });
    });
  }
});
