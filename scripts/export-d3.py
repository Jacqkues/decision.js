"""vllm-sr/d3-lite (text; Qwen3.5-0.8B: 18 Gated DeltaNet + 6 gated attention layers, code readout) → decision.js.

    python scripts/export-d3.py <out dir> [--q4]       (transformers==5.17.0)

Writes manifest.json, weights.bin, the tokenizer files and LICENSE, plus reference.json: the official runtime
on test/fixtures/prompts.json's requests with the original BF16 weights ("original") and with the quantized
weights in fp32 ("quantized"); null where the runtime refuses a question. Layout (q8 by default):
  - zero-centred RMSNorms folded to plain weights (1 + w); the gated norm of the DeltaNet keeps w;
  - attention: q_proj's interleaved [query 256 | gate 256] per head split and fused as [q (8×256) | k (2×256) |
    v (2×256) | gate (8×256)] = 5120 rows;
  - DeltaNet: [qkv 6144 | z 2048 | b 16 | a 16] padded to 8320 rows; conv taps, A_log, dt_bias, norm in f16;
  - MLP: [gate | up] fused; the embedding quantized (gathered only); the readout [255, 1024] in f32.
The vision tower is not exported."""
import argparse, json
import torch
from transformers import AutoModel
from export_common import Blob

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--q4", action="store_true")
args = ap.parse_args()

m = AutoModel.from_pretrained("vllm-sr/d3-lite", trust_remote_code=True, device="cpu")
rt = m.runtime
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
            ans = m.system_one(state=s, questions={"q": q})["answers"]["q"]
            row.append(None if "error" in ans else probs(q, ans))
        out.append(row)
    return out


original = references()
print("original references done", flush=True)
lm = rt.backbone.language_model.float()
c = lm.config
H, I = c.hidden_size, c.intermediate_size
hd, nh, nkv = c.head_dim, c.num_attention_heads, c.num_key_value_heads
cfg = json.load(open(rt.root / "decision_config.json")) if hasattr(rt, "root") else {}
blob = Blob(args.out, {
    "model": "d3", "hidden": H, "intermediate": I, "layer_types": c.layer_types, "eps": c.rms_norm_eps,
    "heads": nh, "kv_heads": nkv, "head_dim": hd, "rotary_dim": int(hd * c.rope_parameters.get("partial_rotary_factor", 1.0)),
    "rope_theta": c.rope_parameters["rope_theta"],
    "linear_heads": c.linear_num_value_heads, "linear_key_dim": c.linear_key_head_dim, "linear_value_dim": c.linear_value_head_dim,
    "conv_kernel": c.linear_conv_kernel_dim, "linear_rows": 8320,
    "codes": cfg.get("codes"), "temperature": cfg.get("temperature", 1.0),
}, 4 if args.q4 else 8)
one = lambda w: 1.0 + w.float()  # noqa: E731  (zero-centred RMSNorm weight)


with torch.no_grad():
    lm.embed_tokens.weight.copy_(blob.quant("embed", lm.embed_tokens.weight))
    for i, layer in enumerate(lm.layers):
        p = f"layers.{i}."
        blob.f16(p + "input_norm", one(layer.input_layernorm.weight))
        blob.f16(p + "post_norm", one(layer.post_attention_layernorm.weight))
        if c.layer_types[i] == "linear_attention":
            la = layer.linear_attn
            ws = [la.in_proj_qkv.weight, la.in_proj_z.weight, la.in_proj_b.weight, la.in_proj_a.weight]
            n = sum(w.shape[0] for w in ws)
            w = torch.cat(ws + [torch.zeros(8320 - n, H)])
            deq = blob.quant(p + "in", w)
            o = 0
            for lin, wt in zip([la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a], ws):
                lin.weight.copy_(deq[o:o + wt.shape[0]]); o += wt.shape[0]
            blob.f16(p + "conv", la.conv1d.weight.reshape(la.conv1d.weight.shape[0], -1))
            blob.f16(p + "A_log", la.A_log)
            blob.f16(p + "dt_bias", la.dt_bias)
            blob.f16(p + "gnorm", la.norm.weight)
            la.out_proj.weight.copy_(blob.quant(p + "out", la.out_proj.weight))
        else:
            a = layer.self_attn
            qg = a.q_proj.weight.view(nh, 2 * hd, H)
            q, g = qg[:, :hd].reshape(nh * hd, H), qg[:, hd:].reshape(nh * hd, H)
            deq = blob.quant(p + "qkvg", torch.cat([q, a.k_proj.weight, a.v_proj.weight, g]))
            dq, dk, dv, dg = deq[:nh * hd], deq[nh * hd:nh * hd + nkv * hd], deq[nh * hd + nkv * hd:nh * hd + 2 * nkv * hd], deq[nh * hd + 2 * nkv * hd:]
            a.q_proj.weight.copy_(torch.cat([dq.view(nh, hd, H), dg.view(nh, hd, H)], 1).reshape(2 * nh * hd, H))
            a.k_proj.weight.copy_(dk); a.v_proj.weight.copy_(dv)
            blob.f16(p + "q_norm", one(a.q_norm.weight))
            blob.f16(p + "k_norm", one(a.k_norm.weight))
            a.o_proj.weight.copy_(blob.quant(p + "o", a.o_proj.weight))
        mlp = layer.mlp
        deq = blob.quant(p + "w13", torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight]))
        mlp.gate_proj.weight.copy_(deq[:I]); mlp.up_proj.weight.copy_(deq[I:])
        mlp.down_proj.weight.copy_(blob.quant(p + "w2", mlp.down_proj.weight))
    blob.f16("final_norm", one(lm.norm.weight))
    blob.f32("readout", rt.readout.float())
blob.close("vllm-sr/d3-lite", None, {"tokenizer.json": "tokenizer.json", "tokenizer_config.json": "tokenizer_config.json", "LICENSE": "LICENSE", "NOTICE": "NOTICE"})
json.dump({"original": original, "quantized": references()}, open(f"{args.out}/reference.json", "w"))
print("reference.json written")
