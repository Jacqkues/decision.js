"""d1-omni-600M (text) as a static-shape graph for LiteRT (.tflite), for the ML Drift comparison.

    python compare/litert/export.py [--batch 24] [--length 136] [--int8]

Inputs: h float32 [B, L, D] (the token embeddings: ML Drift has no GPU lookup in an fp16 or int8 table, so
the caller gathers rows of compare/litert/embed.f16), pad float32 [B, L] (1 on real tokens), qtype float32
[B, 3] (one-hot). Output: the scorer's logit at every token, float32 [B, L] (the caller reads the option
markers; ML Drift misreads a batched one-hot gather). Before the temperature. The forward is the official encoder.py rewritten without dynamic shapes, boolean masks or
nn.TransformerEncoderLayer; checked against the official model on compare/litert/sequences.json, whose
official probabilities are also written to compare/litert/reference.json."""
import argparse, json, math
import torch
import torch.nn.functional as F
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--length", type=int, default=136)
ap.add_argument("--markers", type=int, default=7)
ap.add_argument("--layers", type=int, default=0, help="debugging: keep only the first N trunk layers")
ap.add_argument("--segments", type=int, default=1, help="split the trunk into N chained graphs (LiteRT.js cannot load one ~600 MB graph)")
ap.add_argument("--quant", choices=["fp32", "fp16", "int8", "int8dyn"], default="fp32", help="weights: fp32, fp16, or int8 weight-only per channel")
args = ap.parse_args()
B, L, K = args.batch, args.length, args.markers

model = AutoModel.from_pretrained("LiquidAI/d1-omni-600M", trust_remote_code=True, dtype=torch.float32).eval()
prompt = __import__(type(model).__module__.rsplit(".", 1)[0] + ".prompt", fromlist=["as_question"])
trunk, head = model.encoder, model.head
NEG = -1e4  # fits in fp16 (ML Drift computes in fp16: -1e9 would be -inf, and 0 · -inf = NaN)


class Static(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk, self.head = trunk, head
        cos, sin = trunk.rope(L, torch.zeros(1))
        self.register_buffer("cos", cos)  # (1, 1, L, 64)
        self.head_qkv = torch.nn.ModuleDict()  # the head's fused q/k/v as plain Linear layers (converted like the others)
        for layer in head.head.layers:
            lin = torch.nn.Linear(*reversed(layer.self_attn.in_proj_weight.shape))
            lin.weight.data, lin.bias.data = layer.self_attn.in_proj_weight.data, layer.self_attn.in_proj_bias.data
            self.head_qkv[str(id(layer))] = lin
        self.register_buffer("sin", sin)

    @staticmethod
    def fc(m, x):
        """A Linear on a 2-D view (ML Drift's int8 FULLY_CONNECTED takes [rows, in])."""
        return m(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], -1)

    def rms(self, norm, x):
        """RMSNorm that survives fp16 (ML Drift's precision): the trunk carries activations of ~2,000 from
        layer 9, whose squares overflow fp16; dividing by the row's max |x| first keeps every square ≤ 1.
        w · x / sqrt(mean(x²) + eps) = w · y / sqrt(mean(y²) + eps / s²) with y = x / s."""
        s = x.abs().amax(-1, keepdim=True) + 1e-6
        y = x / s
        return norm.weight * (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + norm.eps / (s * s)))

    def attention(self, a, x, mask):
        b, length, _ = x.shape
        hd = a.head_dim
        q = self.rms(a.q_layernorm, self.fc(a.q_proj, x).view(b, length, a.heads, hd)).transpose(1, 2)
        k = self.rms(a.k_layernorm, self.fc(a.k_proj, x).view(b, length, a.kv_heads, hd)).transpose(1, 2)
        v = self.fc(a.v_proj, x).view(b, length, a.kv_heads, hd).transpose(1, 2)
        rot = lambda t: torch.cat((-t[..., hd // 2:], t[..., : hd // 2]), dim=-1)  # noqa: E731
        q, k = q * self.cos + rot(q) * self.sin, k * self.cos + rot(k) * self.sin
        g = a.heads // a.kv_heads
        # GQA in 4-D (ML Drift takes at most 4 dimensions and no BROADCAST_TO): the g query heads of a
        # key/value head are stacked on the sequence axis
        q = q.reshape(b, a.kv_heads, g * length, hd)
        s = q @ k.transpose(-1, -2) * hd ** -0.5 + mask
        y = (torch.softmax(s, -1) @ v).reshape(b, a.heads, length, hd)
        return self.fc(a.out_proj, y.transpose(1, 2).reshape(b, length, -1))

    def conv(self, c, x, pad):
        z = self.fc(c.in_proj, x * pad[:, :, None])
        d = z.shape[-1] // 3
        bb, cc, u = z[..., :d], z[..., d:2 * d], z[..., 2 * d:]
        bx = bb * u
        w = c.conv.weight[:, 0, :]
        zero = torch.zeros_like(bx[:, :1])
        prev = torch.cat([zero, bx[:, :-1]], 1)
        nxt = torch.cat([bx[:, 1:], zero], 1)
        y = prev * w[:, 0] + bx * w[:, 1] + nxt * w[:, 2]
        return self.fc(c.out_proj, cc * y)

    def trunk_part(self, h, pad, lo, hi):
        mask = ((1.0 - pad) * NEG)[:, None, None, :]
        for layer in self.trunk.layers[lo:hi]:
            x = self.rms(layer.operator_norm, h)
            x = self.attention(layer.self_attn, x, mask) if layer.is_attention_layer else self.conv(layer.conv, x, pad)
            h = h + x
            m = layer.feed_forward
            n = self.rms(layer.ffn_norm, h)
            h = h + self.fc(m.w2, F.silu(self.fc(m.w1, n)) * self.fc(m.w3, n))
        return h

    def forward(self, h, pad, qtype):
        mask = ((1.0 - pad) * NEG)[:, None, None, :]
        for layer in (self.trunk.layers[:args.layers] if args.layers else self.trunk.layers):
            x = self.rms(layer.operator_norm, h)
            x = self.attention(layer.self_attn, x, mask) if layer.is_attention_layer else self.conv(layer.conv, x, pad)
            h = h + x
            m = layer.feed_forward
            n = self.rms(layer.ffn_norm, h)
            h = h + self.fc(m.w2, F.silu(self.fc(m.w1, n)) * self.fc(m.w3, n))
        return self.finish(h, pad, qtype)

    def finish(self, h, pad, qtype):
        mask = ((1.0 - pad) * NEG)[:, None, None, :]
        h = self.rms(self.trunk.embedding_norm, h)
        h = h + (qtype @ self.head.type_emb.weight)[:, None, :]
        for layer in self.head.head.layers:
            sa = layer.self_attn
            x = layer.norm1(h)
            qkv = self.fc(self.head_qkv[str(id(layer))], x)
            d = h.shape[-1]; nh = sa.num_heads; hd = d // nh; b, length, _ = h.shape
            q, k, v = (qkv[..., i * d:(i + 1) * d].view(b, length, nh, hd).transpose(1, 2) for i in range(3))
            s = q @ k.transpose(-1, -2) * hd ** -0.5 + mask
            y = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(b, length, d)
            h = h + self.fc(sa.out_proj, y)
            h = h + self.fc(layer.linear2, F.relu(self.fc(layer.linear1, layer.norm2(h))))
        # the scorer on every token (ML Drift misreads a batched one-hot gather); the caller reads the markers
        sc = self.head.scorer
        return self.fc(sc[3], F.gelu(self.fc(sc[1], sc[0](h)))).squeeze(-1)  # (B, L)


static = Static().eval()
data = json.load(open("compare/litert/sequences.json"))
seqs = data["sequences"]
QT = {"choice": 0, "score": 1, "noul": 2}


def batch(chunk):
    ids = torch.zeros(B, L, dtype=torch.int32); pad = torch.zeros(B, L); mk = torch.zeros(B, K, L); qt = torch.zeros(B, 3)
    for i, s in enumerate(chunk):
        ids[i, :len(s["ids"])] = torch.tensor(s["ids"]); pad[i, :len(s["ids"])] = 1
        for j, m in enumerate(s["markers"]): mk[i, j, m] = 1
        qt[i, QT[s["type"]]] = 1
    return trunk.embed_tokens(ids.long()), pad, qt


if args.layers:  # a truncated model has no reference: skip straight to the conversion
    seqs_ref = None
# ---- parity of the rewrite against the official forward; the official calibrated probabilities ([yes, no]
# for a noul) of every sequence go to reference.json
if not args.layers:
    qobj = [prompt.as_question(q) for q in data["questions"]]
    with torch.no_grad():
        worst, ref = 0.0, []
        for i in range(0, len(seqs), B):
            chunk = seqs[i:i + B]
            z = static(*batch(chunk))  # (B, L): logits at every token
            raw = model._forward([(None, s["ids"], s["markers"], qobj[(i + j) % 3], False) for j, s in enumerate(chunk)])
            ref += model._forward([(None, s["ids"], s["markers"], qobj[(i + j) % 3], True) for j, s in enumerate(chunk)])
            for j, s in enumerate(chunk):
                p = torch.softmax(z[j, s["markers"]], -1).tolist()
                p = p[::-1] if s["type"] == "noul" else p
                worst = max(worst, max(abs(a - b) for a, b in zip(p, raw[j])))
        print(f"{len(seqs)} sequences: largest probability gap, static rewrite vs official forward {worst:.2e}", flush=True)
    json.dump({"probabilities": ref, "temperatures": model.config.temperatures}, open("compare/litert/reference.json", "w"))

trunk.embed_tokens.weight.detach().half().numpy().tofile("compare/litert/embed.f16")


class Part(torch.nn.Module):
    """Trunk layers [lo, hi) (h, pad → h); the last part also runs the final norm, the head and the scorer."""
    def __init__(self, lo, hi, last):
        super().__init__()
        self.s, self.lo, self.hi, self.last = static, lo, hi, last

    def forward(self, h, pad, qtype=None):
        h = self.s.trunk_part(h, pad, self.lo, self.hi)
        if not self.last:
            return h
        return self.s.finish(h, pad, qtype)

# ---- conversion
import litert_torch
from litert_torch.generative.quantize import quant_recipes
recipe = {"fp32": lambda: None, "fp16": quant_recipes.full_fp16_recipe, "int8": quant_recipes.full_weight_only_recipe, "int8dyn": quant_recipes.full_dynamic_recipe}[args.quant]()
sample = batch(seqs[:B])
n = len(trunk.layers)
cuts = [round(i * n / args.segments) for i in range(args.segments + 1)]
for i in range(args.segments):
    last = i == args.segments - 1
    part = Part(cuts[i], cuts[i + 1], last).eval() if args.segments > 1 else static
    inputs = sample if last or args.segments == 1 else sample[:2]
    if args.segments > 1:
        inputs = (torch.randn(B, L, hidden_size := trunk.embed_tokens.weight.shape[1]),) + tuple(inputs[1:])
    edge = litert_torch.convert(part, inputs, quant_config=recipe)
    tag = f"-part{i}of{args.segments}" if args.segments > 1 else ""
    name = f"compare/litert/d1-omni-b{B}-l{L}-{args.quant}{tag}.tflite"
    edge.export(name)
    print(f"{name}: {__import__('os').path.getsize(name) / 1e6:.0f} MB (layers {cuts[i]}-{cuts[i + 1] - 1}{', head' if last else ''})", flush=True)
