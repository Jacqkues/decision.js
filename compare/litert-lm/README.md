# decision.js vs LiteRT-LM (ML Drift on WebGPU): EmbeddingGemma 2

The same architecture (EmbeddingGemma 2's 24-layer text encoder, 768-dim pooled output), the same Chrome on an
Apple M5 Pro. LiteRT-LM runs Google's own export, `embeddinggemma-2-text-270m.litertlm` (int4 QAT weights,
165 MB), through `@litert-lm/core` 0.18.0. decision.js runs the typed-decision fine-tune of the same model
(`mrm8488/embeddinggemma-2-ft-typed-decision`, q8, 293 MB). Times are per text, in ms.

| Tokens per text | LiteRT-LM, 1 text | LiteRT-LM, 32 texts | decision.js, 1 text | decision.js, 32 texts |
|---|---|---|---|---|
| 128 | 15.6 | 15.1 | **13.6** | **7.6** |
| 512 | 51.3 | 51.0 | **46.0** | **37.8** |
| 1024 | 108.8 | 108.2 | **85.4** | **71.9** |

LiteRT-LM runs one text at a time (`computeEmbeddingBatch` loops), padded to a fixed signature (128, 256, 512,
1024 … tokens). decision.js packs texts on one token axis with no padding, so a batch is one forward pass.

For decisions, the gap widens: the model embeds the state, each question and each option separately, and
decision.js keeps the embeddings of texts it has seen. A schema asked of many states is embedded once, so a
support ticket with 3 questions costs one ~70-token text: **6.0 ms per ticket** (300 tickets ×
3 questions, `test/browser/bench.ts`), against at least 15 ms for one 128-token LiteRT-LM call.

## What ML Drift runs (`capture.ts`)

`capture-run.ts` loads LiteRT-LM in Chrome, wraps `GPUDevice.createShaderModule` and the compute pass, and
writes every WGSL shader it compiles plus the dispatch sequence of one embedding (`wgsl-<len>/`, not committed).
For a 128-token text:

- **62 shaders, 1,065 dispatches per text** (about 44 per layer).
- **f16 storage, f32 math.** Every kernel unpacks `vec2<u32>` with `unpack2x16float`, computes in `f32`
  and packs the result back. No `shader-f16` arithmetic, no subgroups, no matrix units: the int4 weights are
  dequantized to f16 at load, and the linears are generic convolution kernels (`dot(vec4, vec4)`, 2 × 4 outputs
  per thread).
- **Epilogue fusion.** gelu(gate) · up is the epilogue of the gate projection (it reads the up projection's
  output), and gelu(gate) · per-layer embedding the epilogue of that branch's gate (shaders 37 and 39).
- **Unfused attention.** Q·Kᵀ goes through the same matmul kernels, then a softmax kernel (online max and sum
  over 4 slices of the row, merged in shared memory), then P·V.

## What decision.js runs instead (`src/eg2-*.ts`)

- **14 dispatches per layer** instead of 44: q | k | v fused into one matmul; q/k/v RMSNorm and RoPE in one
  kernel; one attention kernel (scores, online softmax and P·V; tiles of keys and values shared in workgroup
  memory by the query heads of a key/value head, vec4 reads); Gemma's post-norm, residual add, layer scalar and
  next pre-norm in one kernel (`SANDWICH`); the per-layer embedding's RMSNorm and GELU gating in one kernel.
- **f16 storage and f16 matrix products with f32 accumulation** (the `matmulDirect` kernels of d1-omni). The
  activations stay under 4,100, well inside f16; every sum of squares, softmax and pooling is in f32. The
  embeddings match PyTorch fp32 at cosine 1.00000.
- **Split-K matmuls for few rows.** At 128 tokens, a 32×64 tile walks all of K alone and the GPU is mostly idle
  (about 120 µs per 512×512 matmul). With 2 or 4 subgroups splitting K and adding their f32 partial sums, the
  same matmul takes about 60 µs; one text went from 24.8 to 13.6 ms.
- **Pooling before projecting.** The model projects every token from 512 to 768 dims and then averages them;
  decision.js averages first, then projects once per text (the same result, since the projection is linear).
- **The head on the host.** The schema-conditioned scorer (1.1M parameters) runs in JavaScript in f32, with the
  projections of each text cached: about 1.5 ms per request.

Tried and dropped: register-blocked attention (several queries per lane) was 2× slower (its per-lane arrays
spill out of registers); attention on matrix units in two passes (statistics, then P·V) was 2× slower
(8-query blocks reload keys and values from global memory, and the two passes cost 1.5× the work).

## Reproduce

```sh
npm i                                           # @litert-lm/core is a dev dependency
curl -L -o compare/litert-lm/embeddinggemma-2-text-270m.litertlm \
  https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/main/embeddinggemma-2-text-270m.litertlm
npm run bench -- litert-lm "len=128"                   # LiteRT-LM: ms per text (also 512, 1024)
npx tsx compare/litert-lm/capture-run.ts 128           # the same, and its WGSL shaders and dispatches in wgsl-128/
npm run fetch-model -- eg2-q8
npm run bench -- embed "n=1,32&di=0&len=128"         # decision.js on texts of the same length
```
