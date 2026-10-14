# decision.js

Run small decision models in the browser with WebGPU. Give it some text or JSON and a few questions
(yes/no, multiple choice, a score); it returns the probability of each answer. One forward pass, no text
generation, no server.

**[Try the playground →](https://jacqkues.github.io/decision/playground)**

```ts
import { Decision } from "decision.js";

const model = await Decision.load("https://huggingface.co/Jacqkues/d3-lite-decisionjs/resolve/main/");
// or …/Jacqkues/embeddinggemma-2-typed-decision-decisionjs/resolve/main/ (293 MB, for workflow states)

const { answers } = await model.systemOne(ticket, {
  topic:   { type: "choice", instructions: "What is the topic?", criteria: { billing: null, delivery: null, bug: null } },
  churn:   { type: "noul",   instructions: "Is the customer about to leave?" },
  urgency: { type: "score",  instructions: "How urgent is it?", criteria: ["low", "medium", "high"] },
});
// answers.topic   → { choice: "delivery", confidence: 0.96, probabilities: { billing: 0.01, … } }
// answers.churn   → { noul: 0.83 }  (probability of yes)
// answers.urgency → { score: 1.16, … }  (expected level, 0 = low)
```

decision.js is not on npm yet: clone it and run `npm run build`, or `npm run bundle` for a single ES module.

## Models

| Model | Size | Time per ticket* | Same answer as the original |
|---|---|---|---|
| [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) | 405 MB (q8) | 34 ms | 63 / 63 |
| [d1-3B](https://huggingface.co/LiquidAI/d1-3B) | 1.5 GB (q4) | 166 ms | 55 / 63 |
| [d3-lite](https://huggingface.co/vllm-sr/d3-lite) | 803 MB (q8) | 95 ms | 56 / 56 |
| [Laya](https://huggingface.co/convaiinnovations/laya) | 447 MB (q8) | 39 ms | 46 / 48 |
| [EmbeddingGemma 2 typed decisions](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision) | 293 MB (q8) | 6 ms | 54 / 54 |

\* 3 questions per support ticket, Apple M5 Pro, Chrome. The parity column is checked against each model's
official PyTorch code on a deliberately hard test set ([`test/browser/parity.ts`](test/browser/parity.ts)).

- **d3-lite** is the most accurate on general reasoning (77.6% on a
  [Decision Index](https://github.com/apolinario/decision-index) sample, as its official runtime).
- On d1-omni, decision.js is about 2× faster than Google's LiteRT.js (34 vs 71 ms per ticket, see
  [`compare/litert`](compare/litert/README.md)).
- d3-lite has an optional `shareState` mode that reads the state once for all the questions: 95 → 58 ms,
  slightly less accurate (76.8%).
- **EmbeddingGemma 2 typed decisions** is the best on workflow states (agent traces, tickets, invoices,
  alerts): 0.761 on [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
  against 0.570 for d1-omni and 0.526 for d3-lite, and the fastest by far (6 ms per ticket). It embeds the
  state, each question and each option separately, and decision.js keeps the embeddings it has computed: once a
  schema has been asked, a request costs one pass over its state. It is weak on general reasoning (31.9% on the
  Decision Index sample, 31.6% for its official code). Against Google's LiteRT-LM running the same
  architecture, decision.js is 1.1-1.3× faster on one text and 1.4-2× on batches (see
  [`compare/litert-lm`](compare/litert-lm/README.md)).
- All the benchmarks, with one command each: [`bench/README.md`](bench/README.md).

## API

```ts
const model = await Decision.load(url, {
  onProgress: (loaded, total) => {},  // download progress, in bytes
  shareState: false,                  // d3-lite only: faster, approximate
  workflow: "",                       // EmbeddingGemma 2 only: the workflow named in the state's text
  matrixUnits: true,                  // use the GPU's matrix units when available
  fetch,                              // e.g. a fetch that caches the weights
});

await model.systemOne(state, { name: question, … });          // → { answers, usage }
await model.systemOneBatch([[state, { name: question }], …]);  // many requests, packed on the GPU
await model.probabilities(state, [question, …]);              // raw probabilities
model.destroy();
```

A **state** is a string or any JSON value. A **question** is one of:

- `{ type: "noul", instructions }`: yes/no. The answer is `noul`, the probability of yes.
- `{ type: "choice", instructions, criteria: { option: description | null } }`: one of 2 or more options.
- `{ type: "score", instructions, criteria: [lowest, …, highest] }`: a level on a scale of 2 to 10.

## Browser support

WebGPU with `shader-f16`: recent Chrome or Edge on a laptop or desktop GPU. d3-lite and EmbeddingGemma 2
also need `subgroups`.
Matrix units (`chromium-experimental-subgroup-matrix`, behind `--enable-unsafe-webgpu` for now) make it about
20% faster; without them decision.js falls back to portable kernels with the same answers.

## Exporting weights

decision.js loads its own format (`manifest.json` + `weights.bin`, read with HTTP Range requests). The
d3-lite and EmbeddingGemma 2 exports are on Hugging Face
([d3-lite](https://huggingface.co/Jacqkues/d3-lite-decisionjs),
[EmbeddingGemma 2](https://huggingface.co/Jacqkues/embeddinggemma-2-typed-decision-decisionjs));
`npm run fetch-model -- d3-lite-q8 eg2-q8` downloads them into `models/`. To export the others from the
original checkpoints (Python, `torch`, `transformers`):

```sh
python scripts/export-omni.py models/d1-omni-600M-q8   # --q4 for the 214 MB version
python scripts/export-3b.py   models/d1-3B-q4
python scripts/export-laya.py models/laya-q8           # pip install laya
python scripts/export-d3.py   models/d3-lite-q8        # transformers==5.17.0
python scripts/export-eg2.py  models/eg2-q8            # transformers==5.19.0; --q4 for 158 MB
```

## Development

```sh
npm install
npm test                                           # prompts, token for token, against each model's own code
npm run fetch-model -- eg2-q8                      # a hosted export, into models/
npm run test:browser -- "model=eg2-q8"             # parity on the GPU, in Chrome
npm run bench -- tickets "model=eg2-q8&prof=1"     # speed, and GPU time per kernel
npx vite                                           # demos: /examples/ and /examples/d3.html
```

Every benchmark, how to reproduce it and its results: [`bench/README.md`](bench/README.md).

## License

Apache-2.0 ([NOTICE](NOTICE)). The models keep their own licenses: d1 is under the LFM Open License v1.0,
d3-lite, Laya and EmbeddingGemma 2 typed decisions are Apache-2.0. Each export copies the model's license next to its weights.
