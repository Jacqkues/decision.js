# decision.js (d1.js) vs LiteRT.js (ML Drift on WebGPU)

The same model (d1-omni-600M, text), the same 300 tickets × 3 questions (900 sequences, 85 to 135 tokens
each), in the same Chrome 155 on an Apple M5 Pro. Both are checked against the official PyTorch model in
fp32.

| Engine | Weights | Chrome with `--enable-unsafe-webgpu` | Stock Chrome | Same answer as PyTorch fp32 | Largest probability gap |
|---|---|---|---|---|---|
| d1.js | q8, 405 MB | **34.2 ms/ticket** (matrix units; 41.0 before the direct matmul) | **41 ms/ticket** (portable kernels; 46.5 before f16 weights) | 894 / 900 | 0.060 |
| LiteRT.js 2.5.3 (ML Drift) | fp16, 628 MB | 71.3 ms/ticket | 71.0 ms/ticket | 899 / 900 | 0.012 |

What the numbers include:

- **Padding.** LiteRT needs static shapes, so its graph runs batches of 24 sequences padded to 136 tokens.
  The real tokens average 104, so about 24% of its work is padding. Without it, ML Drift would be near
  54 ms/ticket, still slower than d1.js. d1.js packs sequences with no padding.
- **Inputs.** LiteRT is also timed while JS builds its inputs (the embedding rows and the padding):
  0.7 ms/ticket.
- **Fidelity.** d1.js runs q8 weights and ML Drift fp16 weights, which explains its closer answers. d1.js
  q8 matches PyTorch running the same q8 weights 900/900 (see the d1.js parity tests).

## What it took to run d1-omni on LiteRT.js

`export.py` rewrites the official forward (`encoder.py`) as a static-shape graph and checks it against the
official model on the 900 sequences (largest probability gap 1e-5). Then `litert-torch` converts it. On
the way, each of these broke the WebGPU path:

1. **No tensor above 4 dimensions and no `BROADCAST_TO`.** GQA is rewritten in 4-D: the query heads of a
   key/value head are stacked on the sequence axis.
2. **No GPU lookup in an fp16 or int8 embedding table**, and LiteRT.js cannot split a graph between CPU
   and GPU. So the embedding is gathered in JS (`embed.f16`).
3. **No single graph of about 600 MB or more** (fp32 or fp16): the WebAssembly runtime aborts while
   compiling. So the trunk is split into 4 chained graphs of 4 layers. The hidden states stay on the GPU
   between graphs.
4. **No int8 weights on WebGPU.** Weight-only int8 per channel fails to initialize. Dynamic int8 works for
   a lone Linear but fails in the full model ("Unable to parse bc coord for BATCH axis"). int4 does not
   convert or crashes. See `repro/`.
5. **A batched one-hot gather (`markers @ h`) reads zeros**, so the scorer runs on every token and JS
   reads the markers.
6. **fp16 overflow.** From layer 9, d1-omni's residual stream carries activations of about 2,000. Their
   squares overflow fp16 in RMSNorm, and ML Drift computes in fp16, so every answer was uniform. RMSNorm is
   rewritten to divide by the row's max |x| first. The mask uses -1e4, since -1e9 becomes -inf in fp16 and
   0 · -inf is NaN. fp32 precision would avoid this, but it runs out of memory (point 3).

## Reproduce

```sh
python compare/litert/export.py --quant fp16 --segments 4   # torch, transformers, litert-torch
npx tsx compare/litert/encode.ts                             # sequences.json (run first)
npm run bench -- litert "tflite=d1-omni-b24-l136-fp16.tflite&parts=4&prec=fp16"
SAFE=1 npm run bench -- litert "tflite=d1-omni-b24-l136-fp16.tflite&parts=4&prec=fp16&mu=0"
```
