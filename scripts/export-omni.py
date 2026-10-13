"""LiquidAI/d1-omni-600M (text) → d1.js weights.

    python scripts/export-omni.py <out dir> [--q4] [--revision REV]

Writes manifest.json, weights.bin, the tokenizer files and LICENSE, plus reference.json: the official
probabilities on test/fixtures/prompts.json's requests, with the original weights ("original") and with
the quantized ones ("quantized", what the engine must reproduce).
  - trunk and head linears in q8 (default) or q4, q/k/v fused per attention layer, w1/w3 per MLP;
  - the embedding quantized the same way (only gathered: d1-omni has no LM head);
  - norms, biases, conv taps, the type embedding and the last scorer row in f16."""
import argparse, json
import torch
from transformers import AutoModel
from export_common import Blob, requests

REPO = "LiquidAI/d1-omni-600M"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--q4", action="store_true")
ap.add_argument("--revision")
args = ap.parse_args()

model = AutoModel.from_pretrained(REPO, revision=args.revision, trust_remote_code=True, dtype=torch.float32).eval()
trunk, head = model.encoder, model.head
cfg = model.config.text_config
reqs = requests()
original = model.probabilities_batch(reqs)

hidden = cfg["hidden_size"]
blob = Blob(args.out, {
    "model": "d1-omni", "hidden": hidden, "intermediate": trunk.layers[0].feed_forward.w1.weight.shape[0],
    "heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"], "head_dim": hidden // cfg["num_attention_heads"],
    "vocab": cfg["vocab_size"], "eps": cfg["norm_eps"], "rope_theta": cfg["rope_theta"], "layer_types": cfg["layer_types"],
    "head_layers": len(head.head.layers), "head_heads": head.head.layers[0].self_attn.num_heads,
    "head_ff": head.head.layers[0].linear1.out_features, "ln_eps": head.head.layers[0].norm1.eps,
    "max_length": model.config.max_length, "temperatures": model.config.temperatures}, 4 if args.q4 else 8)


def linear(name: str, lin: torch.nn.Linear):
    lin.weight.copy_(blob.quant(name, lin.weight))


with torch.no_grad():
    trunk.embed_tokens.weight.copy_(blob.quant("embed", trunk.embed_tokens.weight))
    blob.f16("final_norm", trunk.embedding_norm.weight)
    for i, layer in enumerate(trunk.layers):
        p = f"layers.{i}."
        blob.f16(p + "operator_norm", layer.operator_norm.weight)
        blob.f16(p + "ffn_norm", layer.ffn_norm.weight)
        if layer.is_attention_layer:
            a = layer.self_attn
            q, k = a.q_proj.weight.shape[0], a.k_proj.weight.shape[0]
            deq = blob.quant(p + "qkv", torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]))
            a.q_proj.weight.copy_(deq[:q]); a.k_proj.weight.copy_(deq[q:q + k]); a.v_proj.weight.copy_(deq[q + k:])
            linear(p + "o_proj", a.out_proj)
            blob.f16(p + "q_norm", a.q_layernorm.weight)
            blob.f16(p + "k_norm", a.k_layernorm.weight)
        else:
            c = layer.conv
            linear(p + "in_proj", c.in_proj)
            linear(p + "out_proj", c.out_proj)
            blob.f16(p + "conv", c.conv.weight.reshape(hidden, -1))
        m = layer.feed_forward
        n = m.w1.weight.shape[0]
        deq = blob.quant(p + "w13", torch.cat([m.w1.weight, m.w3.weight]))
        m.w1.weight.copy_(deq[:n]); m.w3.weight.copy_(deq[n:])
        linear(p + "w2", m.w2)
    blob.f16("head.type_emb", head.type_emb.weight)
    for i, layer in enumerate(head.head.layers):
        p = f"head.{i}."
        sa = layer.self_attn
        sa.in_proj_weight.copy_(blob.quant(p + "in_proj", sa.in_proj_weight))
        blob.f16(p + "in_proj_bias", sa.in_proj_bias)
        linear(p + "out_proj", sa.out_proj)
        blob.f16(p + "out_proj_bias", sa.out_proj.bias)
        linear(p + "linear1", layer.linear1)
        blob.f16(p + "linear1_bias", layer.linear1.bias)
        linear(p + "linear2", layer.linear2)
        blob.f16(p + "linear2_bias", layer.linear2.bias)
        for nm in ("norm1", "norm2"):
            blob.f16(p + nm, getattr(layer, nm).weight)
            blob.f16(p + nm + "_bias", getattr(layer, nm).bias)
    sc = head.scorer
    blob.f16("scorer.norm", sc[0].weight)
    blob.f16("scorer.norm_bias", sc[0].bias)
    linear("scorer.fc", sc[1])
    blob.f16("scorer.fc_bias", sc[1].bias)
    blob.f16("scorer.out", sc[3].weight.reshape(-1))
    blob.f16("scorer.out_bias", torch.cat([sc[3].bias, torch.zeros(1)]))  # 2 halves: one aligned word
blob.close(REPO, args.revision)

# the model now holds the dequantized weights
json.dump({"original": original, "quantized": model.probabilities_batch(reqs)}, open(f"{args.out}/reference.json", "w"))
print("reference.json written")
