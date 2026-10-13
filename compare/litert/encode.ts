// Encodes the benchmark tickets with d1.js's d1-omni prompt (one sequence per question) for the LiteRT
// comparison: compare/litert/sequences.json = [{ ids, markers, type }] in ticket order, 3 per ticket.
import { writeFileSync } from "node:fs";
import { encodeOmni, TextTokenizer, type Question } from "../../src";
import { hfJson } from "../../test/hf";
import { questions, tickets } from "../../test/browser/tickets";

const tok = TextTokenizer.from(await hfJson("LiquidAI/d1-omni-600M", "tokenizer.json"), await hfJson("LiquidAI/d1-omni-600M", "tokenizer_config.json"));
const qs = Object.values(questions) as Question[];
const seqs = tickets(300).flatMap((t) => qs.map((q) => ({ ...encodeOmni(tok, t, q), type: q.type })));
writeFileSync("compare/litert/sequences.json", JSON.stringify({ questions: qs, sequences: seqs }));
const lens = seqs.map((s) => s.ids.length);
console.log(`${seqs.length} sequences, length ${Math.min(...lens)}–${Math.max(...lens)}, mean ${(lens.reduce((a, b) => a + b) / lens.length).toFixed(0)}`);
