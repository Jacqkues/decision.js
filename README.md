# decision.js

Small decision models in the browser, on WebGPU: [LiquidAI's d1](https://huggingface.co/LiquidAI)
(d1-omni-600M, d1-3B), [d3-lite](https://huggingface.co/vllm-sr/d3-lite) and
[Laya](https://huggingface.co/convaiinnovations/laya). You give decision.js a state (a support ticket, a form, a log line, any JSON) and typed questions. It answers with
calibrated probabilities, in one forward pass and with no tokens generated. Nothing leaves the page.

**[Try it in the playground](https://jacqkues.github.io/decision/playground)** (d3-lite on your GPU; weights on
[Hugging Face](https://huggingface.co/Jacqkues/d3-lite-decisionjs)).

```ts
import { Decision } from "decision.js";

const d1 = await Decision.load("https://example.com/models/d1-omni-600M-q8/");
const { answers } = await d1.systemOne(ticket, {
  topic: { type: "choice", instructions: "What is the topic?", criteria: { billing: "billing, payment", delivery: "delivery", bug: "technical bug" } },
  churn: { type: "noul", instructions: "Is the customer at risk of leaving?" },
  urgency: { type: "score", instructions: "How urgent is it?", criteria: ["low", "medium", "high"] },
});
// answers.topic   → { type: "choice", choice: "delivery", confidence: 0.997, probabilities: {...} }
// answers.churn   → { type: "noul", noul: 0.975 }                          (P(yes))
// answers.urgency → { type: "score", score: 1.28, confidence: 0.47, probabilities: {...}, legend: {...} }
```

The API mirrors the models' own Python API (`system_one`, `system_one_batch`, `probabilities`). It has
the same question schema, the same prompts token for token, and the same answers.

## Models

| Model | Weights | Download | Matrix units | Portable kernels | Same answer as the original (parity set) |
|---|---|---|---|---|---|
| [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) (text) | **q8** | 405 MB | 34 ms/ticket | 41 ms/ticket | 63 / 63 |
| d1-omni-600M (text) | q4 | 214 MB | 34 ms/ticket | 41 ms/ticket | 45 / 63 |
| [d1-3B](https://huggingface.co/LiquidAI/d1-3B) (text) | q4 | 1.5 GB | 166 ms/ticket | 186 ms/ticket | 55 / 63 |
| [d3-lite](https://huggingface.co/vllm-sr/d3-lite) (text) | q8 | 803 MB | 101 ms/ticket | 152 ms/ticket | 56 / 56 |
| [Laya](https://huggingface.co/convaiinnovations/laya) (English) | q8 | 447 MB | 39 ms/ticket | 52 ms/ticket | 46 / 48 |

The speeds are for 300 made-up support tickets with 3 questions each (a 7-way choice, a noul and a
3-level score), measured with `systemOneBatch` on an Apple M5 Pro in Chrome 155. Each d1-omni ticket is
about 310 tokens; d1-3B reads the state once per ticket, about 170 tokens. The parity set is
deliberately hard (empty, nonsensical and forged states, many low-confidence answers). Where the q4
exports differ from the original, PyTorch with the same q4 weights differs in the same way: the gap comes
from the quantization, not from decision.js. **q8 is the recommended d1-omni export:** it gives the original
model's answers at the same speed, because the engine is compute-bound, not memory-bound (and on matrix units
both are converted to f16 on the GPU at load time, see `directWeights`).

**d1-omni-600M** is a bidirectional encoder with a decision head. It reads the state anew for each
question and scores a marker per option. It is the fast one.
**d1-3B** is a causal LFM2 backbone that reads the answer off the logits at the answer slot. The state is
read once per request and shared by its questions (a tree), and only the options' logits are computed.
It is more accurate on hard questions and about 4× slower. A q8 d1-3B export (3 GB) is not supported yet.
**d3-lite** (vLLM Semantic Router's Decision 3.0, Apache-2.0) is a Qwen3.5-0.8B backbone: 18 Gated
DeltaNet layers (a causal linear-attention recurrence, run as a per-sequence scan with f32 state) and 6
gated attention layers with 256-dim heads, bidirectional within the prompt. Each question is its own chat
prompt, and a 255-way readout scores the options' answer codes at the last token. It is by far the most
accurate here (below), and the slowest per ticket: its prompts are longer, it reads the state once per
question, and it has more weights per token. Its own runtime computes in BF16; decision.js matches it.
`shareState: true` (d3-lite only, off by default) reads each request's system prompt and state once for all
its questions: 95 → 58 ms per 3-question ticket. It is an approximation: the state no longer sees the
question, so answers move. Against the exact mode on the 300 tickets, the 7-way topic kept 300/300 answers,
the yes/no 288/300 and the 3-level urgency 250/300; on the Decision Index sample (only BFCL has several
questions per request) accuracy went from 77.8% to 76.8%, BFCL from 91.5% to 89.1%.
**Laya** (Convai Innovations, Apache-2.0) is a ModernBERT-large encoder (28 layers, a global attention
layer every third layer and a ±64-token window in the others, GeGLU) with the same decision head as
d1-omni. Its own runtime refuses a `null` state and noul criteria keyed other than true/false, and so
does decision.js. Its base checkpoint is much weaker than d1-omni on Decision Index (below).

## Fidelity on Decision Index

[`bench/decision-index`](bench/decision-index) runs a sample of the
[Decision Index](https://github.com/apolinario/decision-index) public suite (60 requests from each of
BFCL tool selection, BANKING77, ANLI, ARC-Challenge, BPoMP and Humicroedit: 360 requests, 465 questions)
through decision.js and through each model's official PyTorch code with its original weights. The suite
is not redistributable: `make-sample.py` builds the sample from a suite rebuilt with the official kit.

| Model | Accuracy, decision.js | Accuracy, official PyTorch | Same answer | Largest probability gap |
|---|---|---|---|---|
| d1-omni-600M q8 (matrix units) | 59.8% | 60.0% | 99.1% | 0.053 |
| d1-omni-600M q8 (portable) | 59.6% | 60.0% | 98.9% | 0.046 |
| d3-lite q8 (matrix units) | 77.8% | 77.6% | 98.9% | 0.189 |
| Laya q8 (matrix units) | 38.9% | 39.1% | 99.4% | 0.063 |

Laya scores 0% on BANKING77, as its own runtime does: its 192-token question budget leaves each of the 77
options 3 tokens, too few to tell them apart.

## Compared with LiteRT.js (ML Drift)

On the same 900 sequences in the same Chrome, decision.js runs 34 ms/ticket (41 in stock Chrome) against
71 ms/ticket for Google's LiteRT.js 2.5.3, whose WebGPU backend is ML Drift. Running d1-omni there needed
a rewritten graph split in 4 parts, fp16 weights and an fp16-safe RMSNorm. See
[compare/litert](compare/litert/README.md).

## Parity with PyTorch

[`test/browser/parity.ts`](test/browser/parity.ts) runs every request of
[`test/fixtures/prompts.json`](test/fixtures/prompts.json) through `Decision.load` → `probabilitiesBatch`. That
is 7 states (text, nested JSON, delimiter injection, empty, `null`, a list, a 3,000-character text) × 9
questions, covering every question type, letter codes, 30 options, structured criteria and truncation.
It compares the answers with the official PyTorch code on the same quantized weights and on the original
weights:

| Export | Kernels | Largest probability gap vs PyTorch (same weights) | Same answer, same weights | Same answer as the original |
|---|---|---|---|---|
| d1-omni-600M q8 | matrix units | 0.014 | 63 / 63 | 63 / 63 |
| d1-omni-600M q8 | portable | 0.004 | 63 / 63 | 63 / 63 |
| d1-omni-600M q4 | matrix units | 0.014 | 63 / 63 | 45 / 63 |
| d1-omni-600M q4 | portable | 0.005 | 63 / 63 | 45 / 63 |
| d1-3B q4 | matrix units | 0.002 | 62 / 63 (one near-tie) | 54 / 63 |
| d1-3B q4 | portable | 0.001 | 63 / 63 | 55 / 63 |
| d3-lite q8 | matrix units | 0.005 | 56 / 56 | 56 / 56 |
| d3-lite q8 | portable | 0.001 | 56 / 56 | 56 / 56 |
| Laya q8 | matrix units | 0.009 | 48 / 48 | 46 / 48 |
| Laya q8 | portable | 0.005 | 48 / 48 | 46 / 48 |

The prompts (tokenizer, rendering, truncation, option codes, readout tokens) are checked token for token
against each model's own prompt code (LiquidAI's `prompt.py`, Laya's `build_sequence`, d3-lite's runtime): `npm test`, 277 cases (each model's encoded questions and the ones its runtime refuses).

## API

```ts
const d1 = await Decision.load(baseUrl, {
  onProgress: (loaded, total) => {},   // bytes of weights
  matrixUnits: true,                   // use the GPU's matrix units when the browser exposes them
  tokensPerPass: 8192,                 // tokens per forward pass (memory bound); 4096 for d1-3B
  shareState: false,                   // d3-lite: read a request's state once for all its questions (faster,
                                       // approximate; see above)
  directWeights: true,                 // weights converted once to f16 at load (8×8 blocks on matrix units,
                                       // vec4 column groups otherwise): faster, twice the GPU memory of q8;
                                       // default when under 1.5 GB
  device,                              // an existing GPUDevice (shader-f16 required)
  fetch,                               // e.g. a caching fetch
});

await d1.systemOne(state, { name: question, ... });                 // → { answers, usage }
await d1.systemOneBatch([[state, { name: question }], ...], { onProgress });
await d1.probabilities(state, [question, ...]);                     // → number[][] ([yes, no] for a noul)
await d1.probabilitiesBatch([[state, [question, ...]], ...]);
d1.destroy();
```

A **state** is a string or any JSON value. Questions follow the Decision Index schema:

- `{ type: "noul", instructions, criteria?: { true, false } }`: a yes/no question. The answer is `noul`, P(yes).
- `{ type: "choice", instructions, criteria: { option: description | null } }`: at least 2 options. The answer is `choice`, `confidence` and `probabilities`.
- `{ type: "score", instructions, criteria: [lowest, ..., highest] }`: 2 to 10 levels. The answer is `score`, the expected level.

Batches pack whole requests into forward passes up to `tokensPerPass` tokens, with no padding. Calls are
queued: one forward pass runs on the GPU at a time.

## Browser support

decision.js needs WebGPU with `shader-f16`. It uses two optional features when they are present:

- `subgroups`, for the tiled attention kernel (Chrome and Edge 134 and later);
- `chromium-experimental-subgroup-matrix`, for the matmul on the GPU's matrix units. Chrome currently
  exposes it only behind `--enable-unsafe-webgpu`.

Without matrix units, decision.js uses portable kernels on the same f16 weights (k-major groups of 4 columns
read as vec4 straight from storage, 4×8 outputs per thread): about 20% slower here (41 instead of 34 ms
per d1-omni ticket), with the same answers.

## Weights

decision.js reads its own format: `manifest.json` (config and tensor offsets), `weights.bin` (fetched with HTTP
Range requests, so the server must support them), the model's `tokenizer.json` and
`tokenizer_config.json`, and its `LICENSE`. To export from the Hugging Face checkpoints (Python with
`torch`, `transformers` 5.15 or later, `huggingface_hub`):

```sh
python scripts/export-omni.py models/d1-omni-600M-q8          # add --q4 for the 214 MB version
python scripts/export-3b.py   models/d1-3B-q4
python scripts/export-laya.py models/laya-q8                  # needs `pip install laya`
python scripts/export-d3.py   models/d3-lite-q8               # needs transformers==5.17.0
```

Each export also writes `reference.json`, the official probabilities on the parity requests, for
`npm run test:browser -- "model=d1-omni-600M-q8"`.

Format: linears are quantized in blocks of 32 along K with f16 scales. q4 uses the MatMulNBits layout
(two codes per byte, zero point 8, scale chosen per block by least squares). q8 uses one byte per weight
(zero point 128, absmax scale). q/k/v and w1/w3 are fused. Norms, biases and conv taps are f16.

## How it works

- **One command buffer per batch.** Many sequences (d1-omni) or trees (d1-3B) are packed on one token
  axis. Per-token metadata tells the attention and convolution kernels which tokens each token may read:
  its own sequence for d1-omni (bidirectional, centred 3-tap conv), and its trunk plus its own branch for
  d1-3B (causal).
- **Kernels** ([`src/kernels.ts`](src/kernels.ts)). Activations are f16 and reductions f32. Weights are
  dequantized in the matmul tile loader, except on matrix units: there, after ML Drift's design, the
  linear weights are converted once at load time to f16 in 8×8 blocks, and the matmul loads both operands
  straight from storage into subgroup matrices (one subgroup per 32×64 output tile, no shared-memory
  tiles), flushing f16 partial sums to f32 every 128 values of K: 17% faster end to end than staging
  dequantized tiles through shared memory. The portable matmul runs 64×128 tiles,
  8×8 outputs per thread, with vector loads from shared memory. Both are generated fully unrolled. Attention is tiled (keys and values
  stream through shared memory, 4 lanes share a query). RMSNorm and LayerNorm are fused with the residual
  add, and the convolution reads the gates in place. On matrix units, the MLP's SiLU-and-multiply runs in the
  epilogue of its w1 | w3 matmul (the weights are reordered at load time so a tile holds both halves).
- **d3-lite's DeltaNet layers in one kernel** ([`src/d3-kernels.ts`](src/d3-kernels.ts)): one workgroup per
  (sequence, head) runs the causal conv, the q/k L2 norms and gates, the gated delta rule (MLX's register
  layout, f32 state) and the gated RMSNorm, chunk by chunk through shared memory. The recurrence takes
  ONNX Runtime's single-reduction form (Sᵀk, Sᵀq and k·q reduced together on the old state, then
  o = dec·Sᵀq + (k·q)·δ and the update), and the conv slides each thread over consecutive tokens with the
  previous inputs in registers, so every input is read once: 8.7 → 6.9 ms of GPU time per ticket. With the
  SwiGLU epilogue and an attention kernel that loads each key/value tile once for 2 query heads, the matmuls
  are about 87% of d3's time, and they run at 6.1-6.2 TFLOPS: 92-94% of the 6.6 TFLOPS that WebGPU's
  subgroup matrices reach on an Apple M5 Pro with nothing but multiply-accumulates
  ([`test/browser/peak.ts`](test/browser/peak.ts)).
- **Only what the answer needs.** d1-omni gathers the hidden states at the option markers before its
  scorer. d1-3B computes logits only for the option tokens, at each branch's last token.

## Development

```sh
npm install
npm test                                              # prompt parity (Node, downloads the tokenizers)
npm run test:browser -- "model=d1-omni-600M-q8"       # model parity in Chrome on the GPU
npx tsx scripts/run-browser.ts bench "model=d1-omni-600M-q8&mu=0"
npx tsx scripts/run-browser.ts bench "model=d3-lite-q8&reps=3&prof=1"   # best of 3 runs, then GPU time per kernel
npx vite                                              # then open /examples/ (demo; ?model=/models/<dir>/)
                                                      # or /examples/d3.html: d3-lite, presets, shareState, throughput
```

## License

decision.js is Apache-2.0 (see [NOTICE](NOTICE) for the derived ONNX Runtime kernel, MIT). The d1 models and
tokenizers are LiquidAI's, under the LFM Open License v1.0; d3-lite (vLLM Semantic Router, built on Qwen3.5) and
Laya (Convai Innovations) are Apache-2.0.
decision.js does not ship them: exports copy the model's `LICENSE` (Laya: its model card) next to its weights.
