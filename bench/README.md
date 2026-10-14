# Benchmarks

Every test and benchmark is a page run in Chrome on the real GPU by `scripts/run-browser.ts`:

```sh
npm install
npx playwright install chrome                       # once, if Chrome is not installed
npm run fetch-model -- eg2-q8 d3-lite-q8           # the hosted exports, into models/
npm run bench -- <page> "model=<dir under models/>&<options>"
```

`fetch-model` knows the exports hosted on Hugging Face (`d3-lite-q8`, `eg2-q8`, `eg2-q4`). d1-omni, d1-3B and
Laya are exported from their original checkpoints (`python scripts/export-*.py`, see the main README).

Numbers below: Apple M5 Pro, Chrome, matrix units on (Chrome runs with `--enable-unsafe-webgpu`; `SAFE=1` runs a
stock Chrome). Long runs throttle: compare kernels with `prof=1` (GPU time per kernel) rather than wall time.

| Page | What it measures | Command | Result |
|---|---|---|---|
| `parity` | the answers against each model's own PyTorch code, on [`test/fixtures/prompts.json`](../test/fixtures/prompts.json) | `npm run bench -- parity "model=eg2-q8"` | 54 / 54, cosine 1.00000 |
| `tickets` | ms per support ticket (3 questions), 300 made-up tickets | `npm run bench -- tickets "model=eg2-q8"` | 6.0 ms |
| `typed-decisions` | accuracy, KL and Brier on the test split of [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (2,000 decisions, read from the datasets server) | `npm run bench -- typed-decisions "model=eg2-q8"` | 0.761 |
| `decision-index` | accuracy and agreement with the official runtime on a Decision Index sample | `npm run bench -- decision-index "model=d3-lite-q8&ref=d3"` | 77.6% |
| `embed` | EmbeddingGemma 2: ms per text (`len`, `n` texts), GPU time per kernel (`prof=1`), one request | `npm run bench -- embed "len=128&n=1,32"` | 13.6 / 7.6 ms |
| `litert` | LiteRT.js (ML Drift) on d1-omni, see [`compare/litert`](../compare/litert/README.md) | see there | 71 ms vs 34 ms |
| `litert-lm` | LiteRT-LM (ML Drift) on EmbeddingGemma 2, see [`compare/litert-lm`](../compare/litert-lm/README.md) | `npm run bench -- litert-lm "len=128"` | 15.6 ms |

Common options: `mu=0` (no matrix units, the portable kernels), `direct=0` (weights left quantized), `prof=1`.

## Results by model

`typed-decisions` (workflow states: agent traces, tickets, invoices, security alerts; EmbeddingGemma 2 was
trained on its train split, the others see it for the first time):

| Model | Accuracy | KL ↓ | Brier ↓ | ms per case |
|---|---|---|---|---|
| EmbeddingGemma 2 typed decisions (q8) | **0.761** | **0.120** | **0.066** | 20 |
| EmbeddingGemma 2 typed decisions (q4) | 0.752 | 0.124 | 0.069 | 20 |
| d1-omni-600M (q8) | 0.570 | 0.350 | 0.173 | 174 |
| d3-lite (q8) | 0.526 | 0.521 | 0.247 | 380 |
| Laya (q8) | 0.360 | 0.560 | 0.313 | 234 |
| prior (ignores the input) | 0.470 | 0.347 | 0.189 | |

`decision-index` (NLI, science questions, intents, tool selection, humor: general reasoning):

| Model | decision.js | Official runtime | Same answer | ms per request |
|---|---|---|---|---|
| d3-lite | **77.6%** | 77.6% | 98.7% | 241 |
| d1-omni-600M | 59.8% | 60.0% | 99.1% | 136 |
| Laya | 38.9% | 39.1% | 99.4% | 76 |
| EmbeddingGemma 2 typed decisions | 31.9% | 31.6% | 99.3% | 83 |

## The Decision Index sample

The suite cannot be redistributed (several of its sources forbid it), so the sample and its references stay out
of git. Rebuild it with the official kit ([apolinario/decision-index](https://github.com/apolinario/decision-index)),
then compute the references with each model's own code:

```sh
python -m decision_index suite rebuild --work <work> --only 1 4 12 20 21 27 --skip-download
python bench/decision-index/make-sample.py <work>     # → bench/decision-index/sample.json
python bench/decision-index/reference.py omni         # → reference-omni.json (d1-omni; 3b for d1-3B)
python scripts/d3-reference.py                        # → reference-d3.json  (transformers==5.17.0)
python scripts/laya-reference.py                      # → reference-laya.json (pip install laya)
python scripts/eg2-reference.py                       # → reference-eg2.json  (transformers==5.19.0)
npm run bench -- decision-index "model=eg2-q8&ref=eg2"
```
