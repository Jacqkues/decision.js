// The System One API over either engine: named questions over a state in, typed answers out.
import { D13BEngine } from "./d1-3b-engine.js";
import { D13BPrompt } from "./d1-3b-prompt.js";
import { requestDevice, softmax, type GpuOptions } from "./gpu.js";
import { D1OmniEngine, type OmniSeq } from "./omni-engine.js";
import { LayaEngine } from "./laya-engine.js";
import { D3Engine } from "./d3-engine.js";
import { encode as encodeD3 } from "./d3-prompt.js";
import { encode as encodeLaya } from "./laya-prompt.js";
import { criterion, encode, temperatureKey } from "./omni-prompt.js";
import { answer, checkQuestion, type Question, type Response } from "./questions.js";
import { TextTokenizer } from "./tokenizer.js";

export type ModelKind = "d1-omni" | "d1-3b" | "laya" | "d3";

export interface LoadOptions extends GpuOptions {
  /** Tokens per forward pass (default 8192 for d1-omni, 4096 for d1-3B): bounds GPU memory. */
  /** d3-lite: read each request's system prompt and state once for all its questions (faster, an
   *  approximation: the state no longer sees the question; off by default). */
  shareState?: boolean;
  tokensPerPass?: number;
}

/** A request: a state (a string or any JSON value) and its questions. */
export type Request<Q> = [state: unknown, questions: Q];

export interface BatchOptions {
  /** Called after each forward pass with the requests done so far. */
  onProgress?: (done: number, total: number) => void;
}

export class D1 {
  private queue: Promise<unknown> = Promise.resolve();
  private prompt3B?: D13BPrompt;

  private constructor(
    readonly kind: ModelKind,
    readonly engine: D1OmniEngine | D13BEngine | LayaEngine | D3Engine,
    readonly tokenizer: TextTokenizer,
    private tokensPerPass: number,
    /** d3-lite: see LoadOptions.shareState; can be switched between calls. */
    public shareState = false,
  ) {}

  /** A model exported by scripts/export-omni.py or scripts/export-3b.py, at `base` (a URL ending in "/"
   *  holding manifest.json, weights.bin, tokenizer.json and tokenizer_config.json). */
  static async load(base: string, opts: LoadOptions = {}): Promise<D1> {
    if (!base.endsWith("/")) base += "/";
    const get = opts.fetch ?? globalThis.fetch.bind(globalThis);
    const manifest = await (await get(`${base}manifest.json`)).json();
    const kind: ModelKind = manifest.config?.model === "laya" ? "laya" : manifest.config?.model === "d3" ? "d3"
      : manifest.config?.model === "d1-3b" || !("head_layers" in (manifest.config ?? {})) ? "d1-3b" : "d1-omni";
    const device = opts.device ?? (await requestDevice());
    opts = { ...opts, device };
    // A GPU that runs out of memory may be lost, after which nothing it was asked to do ever finishes: fail instead.
    const lost = device.lost.then((info) => Promise.reject(new Error(`decision.js: the GPU stopped while loading the model (${info.message || info.reason}); it probably lacks the memory for it`)));
    lost.catch(() => undefined);
    const [engine, tokenizer] = await Promise.race([
      Promise.all([
        kind === "d1-omni" ? D1OmniEngine.load(base, opts) : kind === "laya" ? LayaEngine.load(base, opts) : kind === "d3" ? D3Engine.load(base, opts) : D13BEngine.load(base, opts),
        TextTokenizer.load(base, get),
      ]),
      lost,
    ]);
    return new D1(kind, engine, tokenizer, opts.tokensPerPass ?? (kind === "d1-3b" ? 4096 : 8192), opts.shareState);
  }

  /** Named questions over one state, in one call. */
  async systemOne<N extends string>(state: unknown, questions: Record<N, Question>): Promise<Response<N>> {
    return (await this.systemOneBatch([[state, questions]]))[0]!;
  }

  /** Many `[state, {name: question}]` requests, packed into as few forward passes as fit. */
  async systemOneBatch<N extends string>(requests: Request<Record<N, Question>>[], opts: BatchOptions = {}): Promise<Response<N>[]> {
    const done = await this.run(requests.map(([s, qs]) => [s, Object.values(qs) as Question[]]), opts);
    const legend = this.kind === "d1-3b" ? undefined : criterion;
    return requests.map(([, qs], i) => {
      const names = Object.keys(qs) as N[];
      const answers = Object.fromEntries(names.map((n, j) => [n, answer(qs[n], done[i]!.probs[j]!, legend)])) as Response<N>["answers"];
      return { answers, usage: { input_tokens: done[i]!.tokens, output_tokens: 0 as const } };
    });
  }

  /** Each question's distribution over its options ([yes, no] for a noul). */
  async probabilities(state: unknown, questions: Question[]): Promise<number[][]> {
    return (await this.run([[state, questions]], {}))[0]!.probs;
  }

  /** `probabilities` for many `[state, questions]` requests. */
  async probabilitiesBatch(requests: Request<Question[]>[], opts: BatchOptions = {}): Promise<number[][][]> {
    return (await this.run(requests, opts)).map((r) => r.probs);
  }

  /** Frees the GPU weights. */
  destroy() {
    this.engine.destroy();
  }

  /** One call at a time on the GPU (the engines reuse their activation buffers). */
  private run(requests: Request<Question[]>[], opts: BatchOptions) {
    requests.forEach(([, qs]) => qs.forEach(checkQuestion));
    const job = this.queue.then(() => (this.kind === "d1-3b" ? this.run3B(requests, opts) : this.kind === "d3" ? this.runD3(requests, opts) : this.runEncoder(requests, opts)));
    this.queue = job.catch(() => undefined);
    return job;
  }

  /** The encoder models (d1-omni, Laya): one sequence per question, logits at the option markers. */
  private async runEncoder(requests: Request<Question[]>[], opts: BatchOptions) {
    const engine = this.engine as D1OmniEngine | LayaEngine;
    const enc = engine instanceof LayaEngine
      ? (s: unknown, q: Question) => encodeLaya(this.tokenizer, s, q, engine.m.config.max_len, engine.m.config.head_max_len)
      : (s: unknown, q: Question) => encode(this.tokenizer, s, q);
    const seqs = requests.map(([s, qs]) => qs.map((q): OmniSeq => ({ ...enc(s, q), type: q.type })));
    const out = requests.map((_, i) => ({ probs: [] as number[][], tokens: seqs[i]!.reduce((a, x) => a + x.ids.length, 0) }));
    // whole requests per pass, up to the token budget
    for (let i = 0; i < requests.length;) {
      let j = i, used = 0;
      do used += out[j]!.tokens, j++; while (j < requests.length && used + out[j]!.tokens <= this.tokensPerPass);
      const flat = seqs.slice(i, j).flat();
      const { logits } = await engine.logits(flat);
      let k = 0;
      for (let r = i; r < j; r++) {
        out[r]!.probs = requests[r]![1].map((q) => {
          const z = logits[k++]!, t = engine.m.config.temperatures[temperatureKey(q)] ?? engine.m.config.temperatures[q.type] ?? 1;
          const p = softmax(z.map((v) => v / t));
          return q.type === "noul" ? p.reverse() : p; // the model reads a noul as [false, true]
        });
      }
      opts.onProgress?.(j, requests.length);
      i = j;
    }
    return out;
  }

  /** d3-lite: one sequence per question, the answer codes read at its last token. */
  private async runD3(requests: Request<Question[]>[], opts: BatchOptions) {
    const engine = this.engine as D3Engine, c = engine.m.config;
    const seqs = requests.map(([s, qs]) => qs.map((q) => encodeD3(this.tokenizer, s, q, c.codes)));
    const out = requests.map((_, i) => ({ probs: [] as number[][], tokens: seqs[i]!.reduce((a, x) => a + x.ids.length, 0) }));
    for (let i = 0; i < requests.length;) {
      let j = i, used = 0;
      do used += out[j]!.tokens, j++; while (j < requests.length && used + out[j]!.tokens <= this.tokensPerPass);
      const { logits } = await engine.logits(seqs.slice(i, j).flat(), [], this.shareState ? seqs.slice(i, j).map((g) => g.length) : undefined);
      let k = 0;
      for (let r = i; r < j; r++) {
        out[r]!.probs = requests[r]![1].map((q) => {
          const p = softmax(logits[k++]!.map((v) => v / c.temperature));
          return q.type === "noul" ? p.reverse() : p; // the readout lists a noul as [false, true]
        });
      }
      opts.onProgress?.(j, requests.length);
      i = j;
    }
    return out;
  }

  private async run3B(requests: Request<Question[]>[], opts: BatchOptions) {
    const engine = this.engine as D13BEngine;
    const prompt = (this.prompt3B ??= new D13BPrompt(this.tokenizer)); // keeps its option-code caches
    const trees = requests.map(([s, qs]) => ({ trunk: prompt.trunk(s), branches: qs.map((q) => prompt.branch(q)), groups: qs.map((q) => prompt.readoutIds(q)) }));
    const size = (t: (typeof trees)[number]) => t.trunk.length + t.branches.reduce((a, b) => a + b.length, 0);
    const out = trees.map((t) => ({ probs: [] as number[][], tokens: size(t) }));
    for (let i = 0; i < trees.length;) {
      let j = i, used = 0;
      do used += out[j]!.tokens, j++; while (j < trees.length && used + out[j]!.tokens <= this.tokensPerPass);
      const { scores } = await engine.run(trees.slice(i, j));
      scores.forEach((perQ, r) => (out[i + r]!.probs = perQ.map(softmax)));
      opts.onProgress?.(j, trees.length);
      i = j;
    }
    return out;
  }
}
