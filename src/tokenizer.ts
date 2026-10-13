// The model's own tokenizer (tokenizer.json + tokenizer_config.json, shipped next to the weights), through
// Hugging Face's tokenizers.js.
import { Tokenizer } from "@huggingface/tokenizers";

export class TextTokenizer {
  private constructor(private tok: Tokenizer, readonly bos: string, private config: Record<string, unknown>) {}

  static async load(base: string, get: typeof fetch = globalThis.fetch.bind(globalThis)): Promise<TextTokenizer> {
    const [json, config] = await Promise.all([get(`${base}tokenizer.json`).then((r) => r.json()), get(`${base}tokenizer_config.json`).then((r) => r.json())]);
    return TextTokenizer.from(json, config);
  }

  static from(json: object, config: Record<string, unknown>): TextTokenizer {
    const bos = config.bos_token;
    return new TextTokenizer(new Tokenizer(json, config), typeof bos === "string" ? bos : ((bos as { content?: string } | undefined)?.content ?? ""), config);
  }

  /** Token ids of `text`, without special tokens added (special tokens written in the text still match). */
  encode(text: string): number[] {
    return this.tok.encode(text, { add_special_tokens: false }).ids;
  }

  /** A special token of tokenizer_config.json by role ("cls_token", "sep_token", "mask_token"…). */
  special(role: string): string {
    const t = this.config[role];
    const text = typeof t === "string" ? t : (t as { content?: string } | undefined)?.content;
    if (!text) throw new Error(`decision.js: the tokenizer has no ${role}`);
    return text;
  }

  /** The id of one token, e.g. "<|mask|>". */
  id(token: string): number {
    const id = this.tok.token_to_id(token);
    if (id === undefined) throw new Error(`decision.js: the tokenizer has no ${token}`);
    return id;
  }
}
