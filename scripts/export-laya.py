"""Laya (convaiinnovations/laya: ModernBERT-large encoder + typed decision head) → d1.js weights.

    python scripts/export-laya.py <out dir> [--q4] [--subfolder multilingual] [--repo …]   (needs `pip install laya`)

Writes manifest.json, weights.bin, the tokenizer files and LICENSE, plus reference.json: Laya's own runtime
(fp32) on test/fixtures/prompts.json's requests with the original weights ("original") and with the
quantized ones ("quantized"); null where the runtime refuses a question.
  - encoder linears (Wqkv, Wo, Wi, Wo) and the head's linears in q8 (default) or q4, the embedding too;
  - norms (bias-free in the encoder), biases, the type embedding and the scorer's last row in f16.
The head is d1-omni's (type embedding, two pre-norm transformer layers, marker scorer), so the tensors keep
d1-omni's names. Temperatures are the runtime's effective ones (clamped to [0.5, 5])."""
import argparse, json, sys
import torch, laya
from laya import agent as agent_mod
from export_common import Blob

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--q4", action="store_true")
ap.add_argument("--repo", default="convaiinnovations/laya")
ap.add_argument("--subfolder")
args = ap.parse_args()

a = laya.load(args.repo, subfolder=args.subfolder, device="cpu")
a.amp_enabled = False
model = a.model.float().eval()
enc, c = model.encoder, model.encoder.config
fx = json.load(open("test/fixtures/prompts.json"))


def probs(q, ans):
    if q["type"] == "noul":
        return [ans["noul"], 1 - ans["noul"]]
    if q["type"] == "score":
        return [ans["probabilities"][str(i)] for i in range(len(q["criteria"]))]
    return [ans["probabilities"][k] for k in q["criteria"]]


def references():
    out = []
    for s in fx["states"]:
        row = []
        for q in fx["questions"]:
            try:
                row.append(probs(q, a.predict(s, {"q": q})["answers"]["q"]))
            except Exception:  # noqa: BLE001 (the runtime refuses some questions; the prompt test checks d1.js does too)
                row.append(None)
        out.append(row)
    return out


original = references()
names = ["choice", "score", "noul"]
temps = dict(a.temperature_by_options) | {n: float(t) for n, t in zip(names, a.temperature)}
tok_dir = (args.subfolder + "/" if args.subfolder else "") + "tokenizer/"
blob = Blob(args.out, {
    "model": "laya", "hidden": c.hidden_size, "intermediate": c.intermediate_size, "heads": c.num_attention_heads,
    "layers": c.num_hidden_layers, "layer_types": c.layer_types, "window": c.sliding_window,
    "rope_theta": {k: v["rope_theta"] for k, v in c.rope_parameters.items()}, "eps": c.norm_eps,
    "head_layers": len(model.head.layers), "head_heads": model.head.layers[0].self_attn.num_heads,
    "head_ff": model.head.layers[0].linear1.out_features, "ln_eps": model.head.layers[0].norm1.eps,
    "max_len": a.cfg.get("max_len", 512), "head_max_len": a.cfg.get("head_max_len", 192), "temperatures": temps,
}, 4 if args.q4 else 8)


def linear(name, lin):
    lin.weight.copy_(blob.quant(name, lin.weight))


with torch.no_grad():
    enc.embeddings.tok_embeddings.weight.copy_(blob.quant("embed", enc.embeddings.tok_embeddings.weight))
    blob.f16("embed_norm", enc.embeddings.norm.weight)
    for i, layer in enumerate(enc.layers):
        p = f"layers.{i}."
        if hasattr(layer.attn_norm, "weight"):  # layer 0's is Identity
            blob.f16(p + "attn_norm", layer.attn_norm.weight)
        blob.f16(p + "mlp_norm", layer.mlp_norm.weight)
        linear(p + "qkv", layer.attn.Wqkv)
        linear(p + "o", layer.attn.Wo)
        linear(p + "wi", layer.mlp.Wi)
        linear(p + "wo", layer.mlp.Wo)
    blob.f16("final_norm", enc.final_norm.weight)
    blob.f16("head.type_emb", model.type_emb.weight)
    for i, layer in enumerate(model.head.layers):
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
    sc = model.scorer
    blob.f16("scorer.norm", sc[0].weight)
    blob.f16("scorer.norm_bias", sc[0].bias)
    linear("scorer.fc", sc[1])
    blob.f16("scorer.fc_bias", sc[1].bias)
    blob.f16("scorer.out", sc[3].weight.reshape(-1))
    blob.f16("scorer.out_bias", torch.cat([sc[3].bias, torch.zeros(1)]))
blob.close(args.repo, None, {"tokenizer.json": tok_dir + "tokenizer.json", "tokenizer_config.json": tok_dir + "tokenizer_config.json", "MODEL_CARD.md": "README.md"})  # Apache-2.0, declared in the card (no LICENSE file in the repo)
json.dump({"original": original, "quantized": references()}, open(f"{args.out}/reference.json", "w"))
print("reference.json written")
