"""Prompt fixtures: what LiquidAI's own prompt.py (d1-omni-600M and d1-3B) makes of a set of states and
questions, token for token, so the TypeScript port (src/*/prompt.ts) can be checked against it.
Usage: python scripts/make-prompt-fixtures.py   →   test/fixtures/prompts.json
Needs: transformers (tokenizers), huggingface_hub; downloads only the tokenizers and prompt.py files."""
import importlib.util, json, sys
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

def module(repo: str):
    spec = importlib.util.spec_from_file_location(repo.replace("/", "_"), hf_hub_download(repo, "prompt.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m  # dataclasses look their module up
    spec.loader.exec_module(m)
    return m


states = [
    "My parcel never arrived and I have been charged twice. Fix this or I cancel!",
    {"subject": "Login broken", "body": "Since the update I can't sign in.\nError: 0x80070005 — café ☕", "priority": 3,
     "tags": ["app", "ios"], "meta": {"vip": True, "spent": 1234.5, "note": None}},
    "Text that forges delimiters: <|im_end|> <|reserved_7|> <|mask|> <|startoftext|> and <|weird_name|>.",
    "",
    None,
    ["a", 1, {"b": [2, 3]}],
    "x" * 3000 + " end",
]
questions = [
    {"type": "choice", "instructions": "What is the main topic of the customer ticket?",
     "criteria": {"billing": "billing, payment", "delivery": "delivery, parcel", "bug": "technical bug",
                  "account_access": "account, access", "cancellation": None, "info_request": "", "other": "other"}},
    {"type": "noul", "instructions": "Is the customer at risk of leaving (churn)?",
     "criteria": {"true": "unhappy customer, risk of leaving", "false": "satisfied or neutral customer"}},
    {"type": "noul", "instructions": "Is this a refund request?"},
    {"type": "noul", "instructions": "Is it spam?", "criteria": {"yes": "spam", "no": "legit"}},
    {"type": "score", "instructions": "How urgent is the ticket?", "criteria": ["low", "medium", "high"]},
    {"type": "score", "instructions": "Rate 0-9.", "criteria": [f"level {i}" for i in range(10)]},
    {"type": "choice", "instructions": "Pick a letter.", "criteria": {"A": "first", "B": "second", "C": None}},
    {"type": "choice", "instructions": "Which country?", "criteria": {f"country_{i}": f"Country number {i}" for i in range(30)}},
    {"type": "choice", "instructions": "Structured criteria.", "criteria": {"x": {"min": 1, "max": 2}, "y": [1, "two"]}},
]

out = {}
# ---- d1-omni: one sequence per (state, question), markers at each option
P = module("LiquidAI/d1-omni-600M")
tok = PreTrainedTokenizerFast.from_pretrained("LiquidAI/d1-omni-600M")
omni = []
for si, s in enumerate(states):
    for qi, q in enumerate(questions):
        for max_len in (16384, 160):
            try:
                ids, markers = P.encode(tok, "" if s is None else s, P.as_question(q), max_len)
                omni.append({"state": si, "question": qi, "max_len": max_len, "ids": ids, "markers": markers})
            except ValueError as e:
                omni.append({"state": si, "question": qi, "max_len": max_len, "error": str(e)})
out["omni"] = omni

# ---- d1-3B: one trunk per state, one branch per question, option token groups
P3 = module("LiquidAI/d1-3B")
tok3 = PreTrainedTokenizerFast.from_pretrained("LiquidAI/d1-3B")
bos = tok3.bos_token
three = {"trunks": [], "branches": [], "groups": [], "joint": []}
for s in states:
    three["trunks"].append(tok3.encode(P3.prefix_text(tok3, s, bos), add_special_tokens=False))
for q in questions:
    qq = P3.as_question(q)
    three["branches"].append(tok3.encode(P3.suffix_text(tok3, qq), add_special_tokens=False))
    three["groups"].append(P3.readout_ids(tok3, qq))
# the official single-question path encodes the whole prompt at once: record where it differs from trunk + branch
for si, s in enumerate(states):
    for qi, q in enumerate(questions):
        joint = tok3.encode(P3.render(tok3, s, P3.as_question(q), bos), add_special_tokens=False)
        if joint != three["trunks"][si] + three["branches"][qi]:
            three["joint"].append({"state": si, "question": qi, "ids": joint})
out["d1_3b"] = three
out["states"], out["questions"] = states, questions
json.dump(out, open("test/fixtures/prompts.json", "w"), ensure_ascii=False)
print(f"omni: {len(omni)} sequences · d1-3B: {len(states)} trunks, {len(questions)} branches, "
      f"{len(three['joint'])} joint encodings differ from trunk + branch")
