"""Minimal LiteRT models: one Linear(1024 → 1024) on [64, 1024], in every weight format litert-torch offers."""
import torch, litert_torch
from litert_torch.generative.quantize import quant_recipes
from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity as G
lin = torch.nn.Sequential(torch.nn.Linear(1024, 1024, bias=False)).eval()
recipes = {
    "fp32": lambda: None, "fp16": lambda: quant_recipes.full_fp16_recipe(),
    "int8-weight-channel": lambda: quant_recipes.full_weight_only_recipe(),
    "int8-weight-block32": lambda: quant_recipes.full_weight_only_recipe(weight_dtype=Dtype.INT8, granularity=G.BLOCKWISE_32),
    "int8-dynamic-channel": lambda: quant_recipes.full_dynamic_recipe(),
    "int4-weight-block32": lambda: quant_recipes.full_weight_only_recipe(weight_dtype=Dtype.INT4, granularity=G.BLOCKWISE_32),
    "int4-dynamic-block32": lambda: quant_recipes.full_dynamic_recipe(weight_dtype=Dtype.INT4, granularity=G.BLOCKWISE_32),
}
for name, recipe in recipes.items():
    try:
        litert_torch.convert(lin, (torch.randn(64, 1024),), quant_config=recipe()).export(f"compare/litert/repro/linear-{name}.tflite")
        print(name, "converted")
    except Exception as e:
        print(name, "does not convert:", str(e)[:100])
