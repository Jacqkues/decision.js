"""LiquidAI/d1-3B (text) → d1.js weights.

    python scripts/export-3b.py <out dir> [--revision REV]

Writes manifest.json, weights.bin, the tokenizer files and LICENSE, plus reference.json: the official
probabilities on test/fixtures/prompts.json's requests, with the original weights ("original") and with
the quantized ones ("quantized", what the engine must reproduce).
  - linears in q4, q/k/v fused per attention layer, w1/w3 per MLP;
  - the embedding in q4 (also the tied LM head: the engine reads only the options' rows of it);
  - norms and conv taps in f16. The vision tower is not exported."""
import argparse, json
import torch
from transformers import AutoModel
from export_common import Blob, requests

REPO = "LiquidAI/d1-3B"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--revision")
args = ap.parse_args()

model = AutoModel.from_pretrained(REPO, revision=args.revision, trust_remote_code=True, dtype=torch.float32).eval()
lm = model.model.language_model
cfg = model.config.text_config
reqs = requests()
original = model.engine.probabilities_batch(reqs)
print("original references done", flush=True)

H, I = cfg.hidden_size, cfg.intermediate_size
blob = Blob(args.out, {"model": "d1-3b", "hidden": H, "intermediate": I, "heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads,
                       "head_dim": H // cfg.num_attention_heads, "vocab": cfg.vocab_size, "eps": cfg.norm_eps,
                       "rope_theta": cfg.rope_parameters["rope_theta"], "layer_types": cfg.layer_types}, 4)
with torch.no_grad():
    lm.embed_tokens.weight.copy_(blob.quant("embed", lm.embed_tokens.weight))
    blob.f16("final_norm", lm.embedding_norm.weight)
    for i, layer in enumerate(lm.layers):
        p = f"layers.{i}."
        blob.f16(p + "operator_norm", layer.operator_norm.weight)
        blob.f16(p + "ffn_norm", layer.ffn_norm.weight)
        if cfg.layer_types[i] == "conv":
            c = layer.conv
            c.in_proj.weight.copy_(blob.quant(p + "in_proj", c.in_proj.weight))
            c.out_proj.weight.copy_(blob.quant(p + "out_proj", c.out_proj.weight))
            blob.f16(p + "conv", c.conv.weight.reshape(H, -1))
        else:
            a = layer.self_attn
            q, k = a.q_proj.weight.shape[0], a.k_proj.weight.shape[0]
            deq = blob.quant(p + "qkv", torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]))
            a.q_proj.weight.copy_(deq[:q]); a.k_proj.weight.copy_(deq[q:q + k]); a.v_proj.weight.copy_(deq[q + k:])
            a.out_proj.weight.copy_(blob.quant(p + "o_proj", a.out_proj.weight))
            blob.f16(p + "q_norm", a.q_layernorm.weight)
            blob.f16(p + "k_norm", a.k_layernorm.weight)
        m = layer.feed_forward
        deq = blob.quant(p + "w13", torch.cat([m.w1.weight, m.w3.weight]))
        m.w1.weight.copy_(deq[:I]); m.w3.weight.copy_(deq[I:])
        m.w2.weight.copy_(blob.quant(p + "w2", m.w2.weight))
        print(f"layer {i} exported", flush=True)
blob.close(REPO, args.revision)

# the model now holds the dequantized weights (the LM head is tied to the embedding)
json.dump({"original": original, "quantized": model.engine.probabilities_batch(reqs)}, open(f"{args.out}/reference.json", "w"))
print("reference.json written")
