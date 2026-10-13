"""What both exports share: the weights.bin writer, the q4 / q8 / f16 encodings the engines read, the
tokenizer files, and the parity references (the official API on test/fixtures/prompts.json's requests).

Formats (block 32 along K, f16 scales per block):
  q4: MatMulNBits layout, two codes per byte low nibble first, implicit zero point 8; the scale of each
      block is the one of 46 candidates (absmax · f / 7 or / 8) with the least squared error.
  q8: one byte per weight, implicit zero point 128, scale absmax / 127.
"""
import json, os, shutil
import numpy as np
import torch
from huggingface_hub import hf_hub_download

BLOCK = 32


class Blob:
    def __init__(self, out: str, config: dict, bits: int):
        os.makedirs(out, exist_ok=True)
        self.out, self.bits, self.offset = out, bits, 0
        self.f = open(f"{out}/weights.bin", "wb")
        self.manifest = {"format": "d1.js/1", "config": config, "tensors": {}}

    def put(self, raw: bytes) -> int:
        pad = (-self.offset) % 256
        self.f.write(b"\0" * pad)
        self.offset += pad
        start = self.offset
        self.f.write(raw)
        self.offset += len(raw)
        return start

    def quant(self, name: str, w: torch.Tensor, bits: int | None = None) -> torch.Tensor:
        """A linear weight (N, K) quantized; returns the dequantized weight (what the engine computes with)."""
        bits = bits or self.bits
        codes, scales, deq = (quantize8 if bits == 8 else quantize4)(w)
        self.manifest["tensors"][name] = {"dtype": f"q{bits}", "shape": list(w.shape), "offset": self.put(codes.tobytes()), "bytes": codes.nbytes,
                                          "scales": self.put(scales.tobytes()), "scales_bytes": scales.nbytes}
        return deq

    def f16(self, name: str, t: torch.Tensor):
        a = t.detach().float().half().numpy()
        self.manifest["tensors"][name] = {"dtype": "f16", "shape": list(t.shape), "offset": self.put(a.tobytes()), "bytes": a.nbytes}

    def f32(self, name: str, t: torch.Tensor):
        a = t.detach().float().numpy()
        self.manifest["tensors"][name] = {"dtype": "f32", "shape": list(t.shape), "offset": self.put(a.tobytes()), "bytes": a.nbytes}

    def close(self, repo: str, revision: str | None, files: dict | None = None):
        """`files`: {name in the export: path in the repo}, by default the tokenizer files and LICENSE."""
        self.f.close()
        self.manifest["bytes"] = self.offset
        self.manifest["source"] = {"repo": repo, "revision": revision}
        json.dump(self.manifest, open(f"{self.out}/manifest.json", "w"))
        files = files or {n: n for n in ("tokenizer.json", "tokenizer_config.json", "LICENSE")}
        for name, src in files.items():
            dest = f"{self.out}/{name}"
            if os.path.exists(dest):
                os.remove(dest)  # copies from the Hub cache are read-only
            shutil.copyfile(hf_hub_download(repo, src, revision=revision), dest)
        print(f"{self.out}/weights.bin: {self.offset / 1e6:.0f} MB, {len(self.manifest['tensors'])} tensors", flush=True)


def quantize4(w: torch.Tensor, rows: int = 8192):
    if w.shape[0] > rows:  # bounded memory for big matrices (the scale search holds several copies)
        parts = [quantize4(w[i:i + rows]) for i in range(0, w.shape[0], rows)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), torch.cat([p[2] for p in parts])
    n, k = w.shape
    blocks = w.float().reshape(n, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(-1).clamp(min=1e-12)
    best_codes = best_scale = err = None
    for f in torch.linspace(0.6, 1.15, 23):
        for d in (7.0, 8.0):
            scale = (amax * f / d).half().float()
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # an all-zero block: any scale, codes 0
            codes = torch.clamp(torch.round(blocks / scale[..., None]), -8, 7)
            e = ((codes * scale[..., None] - blocks) ** 2).sum(-1)
            if err is None:
                best_codes, best_scale, err = codes, scale, e
            else:
                better = e < err
                best_codes = torch.where(better[..., None], codes, best_codes)
                best_scale = torch.where(better, scale, best_scale)
                err = torch.minimum(e, err)
    deq = (best_codes * best_scale[..., None]).reshape(n, k)
    codes = (best_codes + 8).to(torch.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous().numpy(), best_scale.half().numpy(), deq


def quantize8(w: torch.Tensor):
    n, k = w.shape
    blocks = w.float().reshape(n, k // BLOCK, BLOCK)
    scale = (blocks.abs().amax(-1) / 127).half().float()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # an all-zero block: any scale, codes 0
    codes = torch.clamp(torch.round(blocks / scale[..., None]), -127, 127)
    deq = (codes * scale[..., None]).reshape(n, k)
    return (codes + 128).to(torch.uint8).reshape(n, k).numpy(), scale.half().numpy(), deq


def requests():
    """test/fixtures/prompts.json's states, each with every question."""
    fx = json.load(open(os.path.join(os.path.dirname(__file__), "..", "test", "fixtures", "prompts.json")))
    return [(s, fx["questions"]) for s in fx["states"]]
