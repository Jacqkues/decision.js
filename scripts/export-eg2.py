"""EmbeddingGemma 2 typed decisions (mrm8488/embeddinggemma-2-ft-typed-decision) → decision.js weights.

    python scripts/export-eg2.py <out dir> [--q4]       (transformers==5.19.0; HF_HUB_DISABLE_XET=1 if Xet fails)

Writes manifest.json, weights.bin, the tokenizer files, plus reference.json: the model's own `decide` (fp32) on
test/fixtures/prompts.json's requests with the original weights ("original") and the quantized ones
("quantized"), null where it refuses a request, and the normalized embeddings of a few texts ("embeddings").
  - the text encoder's linears in q8 (default) or q4, the embedding too, with its √hidden scale folded in; the
    per-layer-embedding projection with its 1/√hidden scale folded in, one slice per layer; q | k | v and
    gate | up fused;
  - norms, the per-layer scalars (config) and the embedding projection (pooled on the GPU, then projected) in f16;
  - the decision head (~1.1M parameters, run on the host in f32) in f32, listed under "host".
The model scores option texts against the state and the question, all embedded separately, so the encoder is
the whole GPU work and the head a few small matrix products per question."""
import argparse, json, math
import torch
from transformers import AutoModel
from export_common import Blob

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--q4", action="store_true")
ap.add_argument("--repo", default="mrm8488/embeddinggemma-2-ft-typed-decision")
args = ap.parse_args()

model = AutoModel.from_pretrained(args.repo, trust_remote_code=True, dtype=torch.float32).eval()
mod = __import__(type(model).__module__, fromlist=["render_state"])
tm = model.encoder.language_model
c = tm.config
fx = json.load(open("test/fixtures/prompts.json"))
TYPES = list(model.config.question_types)


def probs(q, ans):
    """The model's answer in d1.js's order: [yes, no] for a noul, else the options' order."""
    p = ans.probabilities
    if q["type"] == "noul":
        return [p["true"], p["false"]]
    if q["type"] == "score":
        return [p[str(i)] for i in range(len(q["criteria"]))]
    return [p[k] for k in q["criteria"]]


def references():
    out = []
    for s in fx["states"]:
        row = []
        for q in fx["questions"]:
            try:
                row.append(probs(q, model.decide(s, {"q": q})["q"]))
            except Exception:  # noqa: BLE001 (it refuses some states: an empty string, a string that starts like JSON)
                row.append(None)
        out.append(row)
    return out


TEXTS = ["hello world", mod.render_state("support", {"subject": "Login broken", "body": "Since the update I can't sign in."}),
         "choice question: What should we do?", "question: Is it spam? | answer: yes"]


def embeddings():
    return model.encode(TEXTS).tolist()


original = references()
sliding = c.per_layer_config["sliding_attention"] if "sliding_attention" in c.per_layer_config else c.per_layer_config[0]
full = c.per_layer_config["full_attention"] if "full_attention" in c.per_layer_config else c.per_layer_config[5]
H, L, PLE = c.hidden_size, c.num_hidden_layers, c.hidden_size_per_layer_input
embed_scale = float(tm.embed_tokens.embed_scale)
ple_scale = tm.ple.per_layer_model_projection_scale
head = model.head
blob = Blob(args.out, {
    "model": "eg2", "hidden": H, "intermediate": c.intermediate_size, "layers": L, "heads": c.num_attention_heads,
    "layer_types": c.layer_types, "window": c.sliding_window, "eps": c.rms_norm_eps,
    "head_dim": {"sliding_attention": sliding.head_dim, "full_attention": full.head_dim},
    "kv_heads": {"sliding_attention": sliding.num_key_value_heads, "full_attention": full.num_key_value_heads},
    "rope_theta": {k: v["rope_theta"] for k, v in c.rope_parameters.items()}, "ple": PLE,
    "layer_scalars": [float(l.layer_scalar) for l in tm.layers], "embedding_dim": c.embedding_dim,
    "prompt": model.config.prompt, "max_len": model.config.max_seq_length or 8192, "embed_dim": model.config.embed_dim,
    "question_types": TYPES, "temperatures": {t: float(v) for t, v in zip(TYPES, model.temperatures)},
    "head": {"width": model.config.head_width, "ln_eps": head.field_norm.eps,
             "prior_scale": math.exp(min(head.log_prior_scale.item(), math.log(100.0))),
             "joint_scale": math.exp(min(head.log_joint_scale.item(), math.log(100.0))),
             "gate": 1 / (1 + math.exp(-head.gate.item())), "residual_bias": head.residual[3].bias.item()},
}, 4 if args.q4 else 8)


def fused(name, linears, scale=1.0):
    """Linears stacked on the output axis, quantized as one; each gets back its share of the dequantized weight."""
    w = torch.cat([l.weight for l in linears]) * scale
    deq = blob.quant(name, w) / scale
    i = 0
    for l in linears:
        n = l.weight.shape[0]
        l.weight.copy_(deq[i:i + n])
        i += n


with torch.no_grad():
    fused("embed", [tm.embed_tokens], embed_scale)
    # the per-layer-embedding projection (hidden → layers · ple), one slice per layer: each layer projects its own
    pw = tm.ple.per_layer_model_projection.weight
    deq = torch.cat([blob.quant(f"layers.{i}.ple_in", pw[i * PLE:(i + 1) * PLE] * ple_scale) for i in range(L)])
    pw.copy_(deq / ple_scale)
    blob.f16("ple.norm", tm.ple.per_layer_projection_norm.weight)
    for i, layer in enumerate(tm.layers):
        p, a = f"layers.{i}.", layer.self_attn
        fused(p + "qkv", [a.q_proj, a.k_proj, a.v_proj])
        blob.f16(p + "q_norm", a.q_norm.weight)
        blob.f16(p + "k_norm", a.k_norm.weight)
        fused(p + "o", [a.o_proj])
        fused(p + "gu", [layer.mlp.gate_proj, layer.mlp.up_proj])
        fused(p + "down", [layer.mlp.down_proj])
        fused(p + "ple_gate", [layer.ple_block.per_layer_input_gate])
        fused(p + "ple_proj", [layer.ple_block.per_layer_projection])
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            blob.f16(p + n, getattr(layer, n).weight)
        blob.f16(p + "ple_norm", layer.ple_block.post_per_layer_input_norm.weight)
    blob.f16("final_norm", tm.norm.weight)
    blob.f16("projection", tm.embedding_projection.weight)
    host = {}
    for n, t in head.named_parameters():
        if t.numel() > 1:  # the scalars are in the config
            blob.f32("head." + n, t)
            host["head." + n] = blob.manifest["tensors"].pop("head." + n)
    blob.manifest["host"] = host
blob.close(args.repo, None, {"tokenizer.json": "tokenizer.json", "tokenizer_config.json": "tokenizer_config.json", "MODEL_CARD.md": "README.md"})  # Apache-2.0 (card)
json.dump({"original": original, "quantized": references(), "texts": TEXTS, "embeddings": embeddings()}, open(f"{args.out}/reference.json", "w"))
print("reference.json written")
