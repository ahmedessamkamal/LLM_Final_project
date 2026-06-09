import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


DATASET_CACHE_PATH = r"I:\project\data\english_dataset"
OUTPUT_DIR         = r"I:\project\output\qwen3-customer-support"
MODEL_NAME = "Qwen/Qwen3-0.6B"   


MAX_SEQ_LEN   = 512
NUM_EPOCHS    = 3
BATCH_SIZE    = 4        
GRAD_ACCUM    = 4        # effective batch = 4 × 4 = 16
LEARNING_RATE = 2e-4
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

print("Loading dataset from local cache...")
ds = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    cache_dir=DATASET_CACHE_PATH
)
print(ds)

dataset = ds["train"]
print(f"\nTotal examples: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"\nSample:\n  instruction: {dataset[0]['instruction']}")
print(f"  response:    {dataset[0]['response'][:80]}...")

# Train / eval split (95% / 5%)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]
print(f"\nTrain: {len(train_dataset)} | Eval: {len(eval_dataset)}")


SYSTEM_PROMPT = (
    "You are a helpful, professional customer support agent. "
    "Answer customer questions clearly, empathetically, and concisely. "
    "If you cannot resolve the issue, escalate politely. "
    "Never make up information you are not sure about."
)

def format_example(example):
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'].strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response'].strip()}<|im_end|>"
    )
    return {"text": text}

print("\nFormatting dataset into ChatML...")
train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
eval_dataset  = eval_dataset.map(format_example,  remove_columns=eval_dataset.column_names)
print(f"Formatted sample:\n{train_dataset[0]['text'][:300]}\n...")


print(f"\nLoading tokenizer from default HF cache...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


print(f"\nLoading Qwen3-0.6B from default HF cache...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model.config.use_cache = False
print(f"Model loaded! Parameters: {model.num_parameters():,}")


lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    save_safetensors=True,
    load_best_model_at_end=True,
    logging_steps=25,
    bf16=True,
    fp16=False,
    report_to="none",
    group_by_length=True,
    dataloader_num_workers=0,  
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
)


trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


print("\n" + "="*55)
print("Starting fine-tuning Qwen3-0.6B...")
print("="*55)
trainer.train(resume_from_checkpoint=True)


print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved successfully!")


from peft import PeftModel

def load_finetuned(base_model_name, adapter_path, model_cache=None):
    tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    mdl = PeftModel.from_pretrained(base, adapter_path)
    mdl.eval()
    return mdl, tok


def generate(model, tokenizer, user_query, max_new_tokens=200):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


print("\n" + "="*55)
print("Testing fine-tuned model...")
print("="*55)

ft_model, ft_tokenizer = load_finetuned(MODEL_NAME, OUTPUT_DIR, None)

test_queries = [
    "I can't log into my account. What should I do?",
    "I want to cancel my subscription immediately.",
    "My order hasn't arrived and it's been 2 weeks.",
    "Can I get a refund for a product I bought last month?",
]

for q in test_queries:
    print(f"\n{'─'*45}")
    print(f"Customer : {q}")
    print(f"Agent    : {generate(ft_model, ft_tokenizer, q)}")


print("\n" + "="*55)
print("BASE MODEL vs FINE-TUNED COMPARISON")
print("="*55)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
base_model.eval()
base_tok = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)
base_tok.pad_token = base_tok.eos_token

comparison_query = "I want to cancel my subscription immediately."
print(f"\nQuery: {comparison_query}")
print(f"\n[BASE MODEL]\n{generate(base_model, base_tok, comparison_query)}")
print(f"\n[FINE-TUNED ]\n{generate(ft_model, ft_tokenizer, comparison_query)}") 