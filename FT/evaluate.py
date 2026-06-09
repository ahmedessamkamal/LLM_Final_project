"""
Evaluation: Base Qwen3-0.6B vs Fine-tuned (QLoRA)
Metrics: ROUGE-L, response length, keyword hit rate
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer

# ─────────────────────────────────────────────
# CONFIG  ← adjust paths if needed
# ─────────────────────────────────────────────
ADAPTER_PATH       = r"I:\project\checkpoint-4700"
MODEL_NAME         = "Qwen/Qwen3-0.6B"
RESULTS_PATH       = r"I:\project\eval_results2.json"

NUM_SAMPLES    = 50    # how many eval examples to run (keep low for speed)
MAX_NEW_TOKENS = 200
SEED           = 42

SYSTEM_PROMPT = (
    "You are a helpful, professional customer support agent. "
    "Answer customer questions clearly, empathetically, and concisely. "
    "If you cannot resolve the issue, escalate politely. "
    "Never make up information you are not sure about."
)

# customer-service keywords that a good agent response should contain
QUALITY_KEYWORDS = [
    "sorry", "apologize", "help", "assist", "resolve", "contact", "support",
    "understand", "issue", "concern", "happy", "please", "thank", "team",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_prompt(instruction: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate(model, tokenizer, instruction: str) -> str:
    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def keyword_hit_rate(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in QUALITY_KEYWORDS if kw in text_lower)
    return hits / len(QUALITY_KEYWORDS)


def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for pred, ref in zip(predictions, references)]
    return float(np.mean(scores))


def avg_length(texts):
    return float(np.mean([len(t.split()) for t in texts]))


def print_separator(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("─" * w)


# ─────────────────────────────────────────────
# LOAD DATASET — Kaludi/Customer-Support-Responses (74 examples)
# ─────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset("Kaludi/Customer-Support-Responses")
eval_split = ds["train"].shuffle(seed=SEED).select(range(min(NUM_SAMPLES, len(ds["train"]))))
instructions = eval_split["query"]
references   = eval_split["response"]
print(f"Evaluating on {len(instructions)} samples from Kaludi/Customer-Support-Responses.\n")

# ─────────────────────────────────────────────
# QUANTIZATION CONFIG (shared)
# ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ─────────────────────────────────────────────
# LOAD TOKENIZER (shared)
# ─────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
# 1. BASE MODEL
# ─────────────────────────────────────────────
print_separator("BASE MODEL")
print(f"Loading {MODEL_NAME} (base)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
base_model.eval()

print("Generating base model responses...")
base_responses = []
for i, instruction in enumerate(instructions):
    resp = generate(base_model, tokenizer, instruction)
    base_responses.append(resp)
    print(f"  [{i+1}/{NUM_SAMPLES}] done", end="\r")
print()

del base_model
torch.cuda.empty_cache()

# ─────────────────────────────────────────────
# 2. FINE-TUNED MODEL
# ─────────────────────────────────────────────
print_separator("FINE-TUNED MODEL")
print(f"Loading {MODEL_NAME} + LoRA adapter from {ADAPTER_PATH}...")
ft_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
ft_model = PeftModel.from_pretrained(ft_base, ADAPTER_PATH, local_files_only=True)
ft_model.eval()

print("Generating fine-tuned model responses...")
ft_responses = []
for i, instruction in enumerate(instructions):
    resp = generate(ft_model, tokenizer, instruction)
    ft_responses.append(resp)
    print(f"  [{i+1}/{NUM_SAMPLES}] done", end="\r")
print()

del ft_model, ft_base
torch.cuda.empty_cache()

# ─────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────
print_separator("COMPUTING METRICS")

print("ROUGE-L...", end=" ", flush=True)
base_rouge = compute_rouge_l(base_responses, references)
ft_rouge   = compute_rouge_l(ft_responses,   references)
print("done")

base_kw  = float(np.mean([keyword_hit_rate(r) for r in base_responses]))
ft_kw    = float(np.mean([keyword_hit_rate(r) for r in ft_responses]))
base_len = avg_length(base_responses)
ft_len   = avg_length(ft_responses)

# ─────────────────────────────────────────────
# 4. PRINT RESULTS TABLE
# ─────────────────────────────────────────────
print_separator("RESULTS")
print(f"{'Metric':<28} {'Base':>10} {'Fine-tuned':>12} {'Winner':>10}")
print("─" * 62)

def row(name, base_val, ft_val, higher_is_better=True):
    if higher_is_better:
        winner = "Fine-tuned" if ft_val > base_val else "Base"
    else:
        winner = "Fine-tuned" if ft_val < base_val else "Base"
    print(f"{name:<28} {base_val:>10.4f} {ft_val:>12.4f} {winner:>10}")

row("ROUGE-L",             base_rouge, ft_rouge)
row("Keyword hit rate",    base_kw,    ft_kw)
row("Avg response length", base_len,   ft_len, higher_is_better=False)

# ─────────────────────────────────────────────
# 5. SIDE-BY-SIDE SAMPLES
# ─────────────────────────────────────────────
print_separator("SAMPLE COMPARISONS (first 3)")
for i in range(min(3, NUM_SAMPLES)):
    print(f"\n[{i+1}] Customer: {instructions[i][:100]}")
    print(f"    Reference : {references[i][:120]}...")
    print(f"    Base      : {base_responses[i][:120]}...")
    print(f"    Fine-tuned: {ft_responses[i][:120]}...")

# ─────────────────────────────────────────────
# 6. SAVE FULL RESULTS TO JSON
# ─────────────────────────────────────────────
results = {
    "num_samples": NUM_SAMPLES,
    "metrics": {
        "base":       {"rouge_l": base_rouge, "keyword_hit_rate": base_kw, "avg_response_len": base_len},
        "fine_tuned": {"rouge_l": ft_rouge,   "keyword_hit_rate": ft_kw,   "avg_response_len": ft_len},
    },
    "samples": [
        {"instruction": instructions[i], "reference": references[i],
         "base_response": base_responses[i], "ft_response": ft_responses[i]}
        for i in range(NUM_SAMPLES)
    ],
}
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nFull results saved to: {RESULTS_PATH}")
print_separator()
