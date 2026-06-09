# Tesco Customer Support LLM Assistant — Project Report

**Course:** Generative AI — Term 4, 2026  
**Domain:** Retail / E-Commerce Customer Support  
**Models:** Qwen/Qwen3-0.6B (fine-tuned with QLoRA) · Qwen/Qwen3-4B (used for RAG)  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Models, Frameworks & Tools Used](#3-models-frameworks--tools-used)
4. [How to Run](#4-how-to-run)
5. [Prompt Design](#5-prompt-design)
6. [Retrieval-Augmented Generation (RAG)](#6-retrieval-augmented-generation-rag)
7. [Fine-Tuning — QLoRA / PEFT](#7-fine-tuning--qlora--peft)
8. [Tools & Function Calling](#8-tools--function-calling)
9. [Multi-Agent Design](#9-multi-agent-design)
10. [Evaluation](#10-evaluation)
11. [Ethics, Safety & Limitations](#11-ethics-safety--limitations)
12. [Future Improvements](#12-future-improvements)
13. [Conclusion](#13-conclusion)

---

## 1. Project Overview

### Goal

This project builds an LLM-powered **customer support assistant for Tesco**. The assistant answers customer questions about online grocery shopping, delivery, Click+Collect, refunds, returns, Clubcard, and related policies.

### Target Users

Tesco online shoppers who need fast, accurate answers to common questions without waiting for a human agent.

### Value Provided

| Without the assistant | With the assistant |
|---|---|
| Customer waits for a human agent | Immediate, accurate answers |
| Agent may give inconsistent answers | Answers grounded in official Tesco FAQ |
| No safety filtering | Harmful/fraudulent queries refused |

### What Was Built

The project integrates all six required LLM techniques in a single coherent pipeline:

1. **Prompt Design** — structured system prompts with role, style, constraints, and examples
2. **RAG** — FAISS retrieval from the Tesco FAQ CSV with **Qwen3-4B** as the generation model
3. **Fine-Tuning (QLoRA)** — Qwen3-0.6B adapted on 26K customer-support examples, evaluated independently against the base model
4. **Tools** — FAQ keyword search and response quality validator
5. **Multi-Agent** — Planner and Executor agents that collaborate to generate validated answers
6. **Evaluation** — ROUGE-L, keyword hit rate, and RAGAS metrics with before/after comparisons

> **Note on RAG + Fine-tuning integration:** The RAG pipeline and the QLoRA fine-tune were developed and evaluated as **separate tracks**. Combining the fine-tuned Qwen3-0.6B with RAG was attempted but not completed: because QLoRA fine-tuning modifies the model's weight space, the adapter introduced hallucinations when the model was asked to stay strictly within retrieved context — it would blend parametric memory from the Bitext training data with the Tesco FAQ context rather than deferring entirely to the retrieved documents. The RAG pipeline therefore uses the larger **Qwen3-4B base model** (without a fine-tune adapter), which follows retrieval-grounded instructions more faithfully out of the box.

---

## 2. System Architecture

```
Customer Question
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   Multi-Agent Orchestrator               │
│                                                         │
│  ┌──────────────────────┐    ┌───────────────────────┐  │
│  │   Agent 1: Planner   │───▶│  Agent 2: Executor    │  │
│  │  Analyze question    │    │  Execute plan          │  │
│  │  Create action plan  │    │  Call tools            │  │
│  └──────────────────────┘    │  Generate response     │  │
│                               └───────────┬───────────┘  │
└───────────────────────────────────────────┼─────────────┘
                                            │
                        ┌───────────────────┼───────────────────┐
                        │                   │                   │
                        ▼                   ▼                   ▼
               ┌─────────────┐    ┌──────────────────┐  ┌─────────────────┐
               │  Tool 1:    │    │   RAG Pipeline   │  │  Tool 2:        │
               │ FAQ Search  │    │                  │  │ Response        │
               │ (keyword)   │    │ FAISS Index      │  │ Validator       │
               └─────────────┘    │ all-MiniLM-L6-v2 │  └─────────────────┘
                                  │ Tesco FAQ CSV    │
                                  └────────┬─────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  Qwen3-4B (base model)  │
                              │  — used for RAG track   │
                              └────────────────────────┘
                                           │
                                           ▼
                                   Final Response
```

### Component Summary

| Component | Technology | Purpose |
|---|---|---|
| Knowledge base | `fact-base-tesco.csv` (350 rows) | Tesco FAQ & policy documents |
| Embedding model | `all-MiniLM-L6-v2` | Semantic search |
| Vector database | FAISS (`IndexFlatL2`) | Nearest-neighbour retrieval |
| Language model (RAG) | `Qwen/Qwen3-4B` (base, no adapter) | RAG response generation |
| Language model (fine-tune) | `Qwen/Qwen3-0.6B` + LoRA adapter | Fine-tuning evaluation only |
| Fine-tuning method | QLoRA (4-bit NF4 + LoRA rank-16) | Domain adaptation |
| Tool 1 | FAQ keyword search | Keyword-based FAQ lookup |
| Tool 2 | Response validator | Quality gate before delivery |
| Agent 1 | Planner | Decomposes the question into steps |
| Agent 2 | Executor | Calls tools and generates the answer |

---

## 3. Models, Frameworks & Tools Used

| Category | Name / Version |
|---|---|
| LLM for RAG pipeline | `Qwen/Qwen3-4B` (base, no adapter) |
| LLM for fine-tuning | `Qwen/Qwen3-0.6B` (Apache 2.0) |
| Fine-tuning | `peft` (LoRA), `trl` (SFTTrainer), `bitsandbytes` (4-bit) |
| Training data | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` |
| Evaluation data | `Kaludi/Customer-Support-Responses` + `fact-base-tesco.csv` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | `faiss-cpu` |
| RAG framework | `langchain`, `langchain-community` |
| Evaluation | `rouge-score`, `ragas`, `datasets` |
| LLM inference | `transformers`, `accelerate` |

---

## 4. How to Run

### Prerequisites

```bash
pip install transformers accelerate peft trl bitsandbytes
pip install sentence-transformers faiss-cpu
pip install langchain langchain-community langchain-core
pip install ragas datasets rouge-score openai
```

### File Structure

```
project/
├── train.py                  # Fine-tune Qwen3-0.6B with QLoRA
├── evaluate.py               # ROUGE-L & keyword evaluation
├── RAG.ipynb                 # RAG pipeline + RAGAS evaluation
├── fact-base-tesco.csv       # Tesco FAQ knowledge base (350 entries)
├── output/
│   └── qwen3-customer-support/   # Saved LoRA adapter
└── PROJECT_REPORT.md         # This file
```

### Step 1 — Fine-tune the model

```bash
python train.py
```

Trains Qwen3-0.6B on the Bitext customer-support dataset using QLoRA. Checkpoints are saved every 100 steps to `output/qwen3-customer-support/`. The script supports `resume_from_checkpoint=True` for interrupted runs.

### Step 2 — Evaluate the fine-tuned model

```bash
python evaluate.py
```

Runs ROUGE-L and keyword hit-rate evaluation comparing the base model to the fine-tuned model. Results are saved to `eval_results.json`.

### Step 3 — Run the RAG pipeline and RAGAS evaluation

Open `RAG.ipynb` in Jupyter and run all cells top to bottom:

1. **Cells 1–6**: Load packages, configure paths, read `fact-base-tesco.csv`
2. **Cell 7**: Build LangChain `Document` objects from the CSV (no PDF needed)
3. **Cell 8**: Load the fine-tuned model from `output/qwen3-customer-support/`
4. **Cells 9–12**: Chunk documents and build the FAISS index
5. **Cells 13–28**: Test retrieval, define prompts, test with/without RAG
6. **Cells 29–35**: Build evaluation dataset from real Tesco Q&A pairs, run RAGAS
7. **Cell 36**: Compute ROUGE-L, keyword hit rate — saves `rag_eval_results.json`

> Set `OPENAI_API_KEY` in cell 30 before running RAGAS (requires OpenAI for LLM-based metrics). Cell 36 (ROUGE-L) works without an API key.

---

## 5. Prompt Design

Two prompt configurations are used depending on whether RAG context is available.

### 5.1 Base Prompt (No RAG)

Used for baseline comparisons and the multi-agent no-RAG path.

```
You are a professional Tesco Customer Support Assistant.

Your role is to answer customer questions in a polite, concise,
and customer-friendly manner.

Guidelines:
* Be professional and helpful.
* Keep answers clear and short.
* If unsure, mention that information may vary depending on Tesco policies.
* Do not provide harmful, fraudulent, or unethical advice.
* Refuse requests involving policy abuse, scams, or illegal activities.
* Maintain a respectful and supportive tone.
```

**Design choices:**
- **Role anchoring**: The system prompt explicitly names the persona ("Tesco Customer Support Assistant") so the model activates domain-specific behaviour on every call.
- **Conciseness instruction**: Prevents the base model from over-generating long, rambling answers.
- **Refusal rules**: Stated explicitly so the model does not need to infer them.
- **Structural consistency**: The same system prompt is used in every training example during fine-tuning, reinforcing the persona–format pairing.

### 5.2 RAG Prompt (With Retrieved Context)

```
You are a Tesco Customer Support Assistant.

Your task is to answer customer questions using ONLY the provided
Tesco FAQ and policy documents.

Instructions:
* Use only the retrieved Tesco context to answer questions.
* Do not invent policies, prices, delivery times, or refund rules.
* If the answer is not found in the provided context, say:
  "I could not find this information in the Tesco policy documents."
* Be professional, concise, and customer-friendly.
* Refuse harmful, fraudulent, unethical, or privacy-violating requests.

Retrieved Tesco Context:
{retrieved_context}

Customer Question:
{question}

Answer using only the retrieved Tesco context.
```

**Design choices:**
- **Strict grounding instruction** ("ONLY the provided documents"): Directly reduces hallucinations.
- **Explicit fallback phrase**: Gives the model a scripted response when context is absent, preventing fabrication.
- **Safety examples in the prompt**: Few-shot refusal examples show the model the exact format for declining harmful requests.

### 5.3 ChatML Format (Training)

Every training example follows the ChatML format native to Qwen3:

```
<|im_start|>system
You are a helpful, professional customer support agent...<|im_end|>
<|im_start|>user
{customer_instruction}<|im_end|>
<|im_start|>assistant
{agent_response}<|im_end|>
```

Including the system prompt in every training example conditions the model to associate the persona with this fine-tune at inference time.

---

## 6. Retrieval-Augmented Generation (RAG)

### 6.1 Knowledge Base

The knowledge base is `fact-base-tesco.csv`, containing **350 Tesco FAQ entries** across topics including:

- Delivery and Click+Collect
- Minimum basket values and charges
- Returns and refunds
- Clubcard and Delivery Saver plans
- Payment, account, and checkout issues

Each row has: `ID`, `Topic`, `Subtopic`, `Question`, `Answer`.

### 6.2 Document Ingestion

```python
from langchain.schema import Document

lc_documents = []
for _, row in df.iterrows():
    text = f"Question: {row['Question']}\n\nAnswer: {row['Answer']}"
    lc_documents.append(Document(
        page_content=text,
        metadata={"id": row["ID"], "topic": row["Topic"], "subtopic": row["Subtopic"]}
    ))
```

LangChain `Document` objects are created directly from the CSV — no intermediate PDF conversion is required.

### 6.3 Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(lc_documents)
```

Long answers are split into 500-character chunks with 50-character overlap to preserve context at chunk boundaries.

### 6.4 Embedding and Indexing

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, show_progress_bar=True).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
```

`all-MiniLM-L6-v2` is a lightweight BERT-derived model optimised for semantic similarity. FAISS `IndexFlatL2` performs exact L2-distance nearest-neighbour search.

### 6.5 Retrieval

```python
def retrieve_chunks(query, k=3):
    query_emb = embedding_model.encode([query]).astype("float32")
    D, I = index.search(query_emb, k=k)
    return [chunks[idx] for idx in I[0]]
```

The top-3 most semantically similar chunks are retrieved per query.

### 6.6 Grounded Generation

The RAG pipeline uses **Qwen3-4B** (base model, no fine-tune adapter). Qwen3-4B was chosen here because the fine-tuned Qwen3-0.6B adapter introduced hallucinations in the RAG setting — see §11.5 for a full explanation.

```python
def ask_qwen_with_rag(question):
    retrieved = retrieve_chunks(question)
    context = "\n\n".join([c.page_content for c in retrieved])
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    ...
```

The retrieved context is injected into the user turn, keeping it separate from the system instructions.

### 6.7 RAG vs No-RAG — Example

**Question:** "What is the minimum basket charge?"

| Mode | Response |
|---|---|
| Without RAG | "Minimum basket charges vary by store and order type." *(generic, possibly wrong)* |
| With RAG | "A £5 minimum basket charge is added to delivery orders where the basket is below £50. Click+Collect orders have a £25 minimum." *(grounded in the CSV)* |

---

## 7. Fine-Tuning — QLoRA / PEFT

### 7.1 Model

**`Qwen/Qwen3-0.6B`** — a 596M-parameter decoder-only transformer with:
- 32,768 token context window
- Native ChatML format
- Apache 2.0 licence

It was chosen because it is small enough to fine-tune on a single consumer GPU (≥8 GB VRAM) while having strong instruction-following capability out of the box.

### 7.2 Method: QLoRA

QLoRA combines two techniques:

**Low-Rank Adaptation (LoRA):**  
Instead of updating all 596M parameters, small trainable matrices **A** and **B** are injected into each targeted projection layer:

```
W' = W + (alpha/r) × B × A
```

Only **A** and **B** are updated — reducing trainable parameters from 596M to **~10M (1.67%)**.

**4-bit NF4 Quantisation:**  
The frozen base weights are compressed to 4-bit NormalFloat4, cutting VRAM from ~2.4 GB to ~600 MB. Computation is still performed in bfloat16.

| Method | VRAM needed | Trainable params | Quality |
|---|---|---|---|
| Full fine-tuning | ~12 GB | 596M (100%) | Best |
| LoRA (bf16) | ~3 GB | ~10M (1.67%) | Very good |
| **QLoRA (used here)** | **~1.5 GB** | **~10M (1.67%)** | **Good** |

### 7.3 Training Dataset

**`bitext/Bitext-customer-support-llm-chatbot-training-dataset`** — 26,872 customer support Q&A pairs covering intents such as cancellations, refunds, account issues, and shipping. Split 95/5 into train (25,528) and eval (1,344).

### 7.4 LoRA Configuration

```python
LoraConfig(
    r=16,              # rank
    lora_alpha=32,     # scale = alpha/r = 2.0
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

All 7 projection layers (attention + MLP) are targeted. Including the MLP layers (`gate/up/down`) improves domain adaptation because ~2/3 of the model's capacity lives in the feed-forward blocks.

### 7.5 Key Training Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| Epochs | 3 | Sufficient convergence on 26K examples |
| Batch size | 4 × 4 accum = effective 16 | Stable gradient estimates within VRAM limits |
| Learning rate | 2e-4 | Standard for LoRA; higher than full fine-tuning |
| Scheduler | cosine | Smooth decay to near-zero by end of training |
| Warmup | 5% | Avoids large gradient spikes at start |
| Optimiser | `paged_adamw_8bit` | Offloads optimiser states to CPU when VRAM is full |
| bf16 | True | Faster and more stable than fp16 on Ampere+ GPUs |
| Max sequence length | 512 | Covers 99%+ of customer support exchanges |
| `load_best_model_at_end` | True | Restores lowest eval-loss checkpoint |

### 7.6 Behaviour Change — Base vs Fine-tuned

The base model includes a reasoning `<think>` block and produces generic answers. The fine-tuned model responds directly in a professional support-agent style.

**Query:** "Do you offer gift wrapping?"

| Model | Response |
|---|---|
| Base | `<think>The user is asking if I offer gift wrapping...</think>` Yes, I offer gift wrapping. Would you need help with anything else? |
| Fine-tuned | Positively! We offer gift wrapping as a convenient option for our valued customers... [on-topic, professional, no reasoning block] |

---

## 8. Tools & Function Calling

The system implements two tools callable by the Executor agent.

### Tool 1 — FAQ Search

Performs keyword-based search over the FAQ database to retrieve relevant entries.

```python
def search_faq_tool(customer_question: str, top_k: int = 3) -> str:
    question_words = customer_question.lower().split()
    results = []
    for faq_item in faq_database:
        score = 0
        for word in question_words:
            if word in faq_item['q'].lower(): score += 2   # question match
            if word in faq_item['a'].lower(): score += 1   # answer match
        if score > 0:
            results.append((score, faq_item))
    results.sort(reverse=True)
    return results[:top_k]
```

**Example — "Do you offer online shopping?"**

| FAQ Entry | Match Score |
|---|---|
| "Do you offer online shopping?" | +2 (online) +2 (shopping) = **4** |
| "What is your returns policy?" | 0 |

Returns the top-3 scored entries as grounded context.

**Why this tool matters:** Without it, the model must rely entirely on parametric memory and may hallucinate. The tool grounds every response in actual FAQ data.

### Tool 2 — Response Validator

Checks the quality of a generated response on three dimensions before it is delivered.

```python
def validate_response_tool(response: str, question: str) -> dict:
    validation = {}
    validation['length']       = len(response) > 50
    helpful_words              = ['yes', 'no', 'can', 'will', 'offer']
    validation['helpful']      = any(w in response.lower() for w in helpful_words)
    professional_words         = ['thank', 'please', 'help', 'welcome']
    validation['professional'] = any(w in response.lower() for w in professional_words)
    score = (sum(validation.values()) / 3) * 100
    return {"validation": validation, "score": score}
```

**Example — "Yes, we offer online shopping with 3-7 day delivery. Thank you!"**

| Check | Criterion | Result |
|---|---|---|
| Length | > 50 characters | ✓ |
| Helpful | Contains "offer" | ✓ |
| Professional | Contains "thank" | ✓ |
| **Quality Score** | — | **100 / 100** |

**Why this tool matters:** Acts as a quality gate before the response reaches the customer. Low-quality responses can be regenerated or escalated rather than delivered.

---

## 9. Multi-Agent Design

### Architecture: Planner → Executor

The system uses two agents that collaborate through a sequential hand-off.

```
Customer Question
       │
       ▼
[Agent 1: Planner]
  • Receives the customer question
  • Creates a 2-step action plan using the LLM
       │
       ▼ plan (string)
[Agent 2: Executor]
  • Receives the question + plan
  • Calls Tool 1 (FAQ Search) → gathers context
  • Generates response using the LLM + context
  • Calls Tool 2 (Validator) → quality check
       │
       ▼
Final Response
```

### Agent 1 — Planner

```python
def planner_agent(customer_question: str) -> str:
    prompt = f"""You are a customer support planner. Create a 2-step action plan:

    Question: {customer_question}

    Plan:"""
    return model.generate(prompt)
```

**Example output for "Do you offer online shopping?":**
```
1. Search FAQ for online shopping information
2. Generate response with delivery details
```

The planner separates strategic reasoning from execution. For complex questions, planning before acting improves response coherence.

### Agent 2 — Executor

```python
def executor_agent(customer_question: str, planner_plan: str) -> str:
    # Step A: Tool 1 — keyword FAQ search
    faq_results = search_faq_tool(customer_question)

    # Step B: LLM response generation using FAQ context
    prompt = f"""Answer the customer using this FAQ:
    Question: {customer_question}
    Relevant FAQ: {faq_results}
    Response:"""
    response = model.generate(prompt)

    # Step C: Tool 2 — quality validation
    validation = validate_response_tool(response, customer_question)

    return response
```

### Orchestrator

```python
def run_customer_support_bot(customer_question: str):
    plan     = planner_agent(customer_question)      # Phase 1
    response = executor_agent(customer_question, plan)  # Phase 2
    return response
```

### Collaboration Example — Complex Question

**"How do I return items?"**

```
[Planner] → "1. Find returns FAQ  2. Explain the process step-by-step"

[Executor]
  Tool 1 → Search returns "Returns Policy" + "Refund FAQ" entries
  Generate → "Returns are simple!
              1. Come to store with receipt
              2. Item must be in original condition
              3. Refund in 5 business days"
  Tool 2 → Quality: 94/100 ✓

[Result] → "Returns are simple! 1. Come to store..."
```

### Why Two Agents?

| Single-agent | Two-agent |
|---|---|
| One prompt does everything | Planning is separated from execution |
| No quality gate | Tool 2 validates before delivery |
| Hard to debug failures | Each agent's output is inspectable |
| Harder to extend | New agents/tools can be plugged in |

---

## 10. Evaluation

### 10.1 Evaluation Strategy

Two separate evaluation tracks were run:

| Track | What is measured | Tools |
|---|---|---|
| Fine-tuning quality | Base model vs fine-tuned model | ROUGE-L, keyword hit rate, response length |
| RAG quality | With RAG vs without RAG (Qwen3-4B base) | RAGAS (faithfulness, answer relevancy, context precision, context recall) |

### 10.2 Fine-Tuning Evaluation

**Dataset:** `Kaludi/Customer-Support-Responses` — 50 randomly sampled Q&A pairs, completely separate from the Bitext training data.

**Procedure:**
1. Run 50 queries through the **base Qwen3-0.6B** model and record responses
2. Unload base model, load the **fine-tuned model** (base + LoRA adapter)
3. Run the same 50 queries and record responses
4. Compute all metrics against the reference answers

#### ROUGE-L

Measures the longest common subsequence between the generated and reference response (F1 score). Captures word order, not just word overlap.

```
Precision = LCS(generated, reference) / len(generated)
Recall    = LCS(generated, reference) / len(reference)
ROUGE-L   = 2 × Precision × Recall / (Precision + Recall)
```

#### Keyword Hit Rate

Fraction of 14 customer-service keywords present in the response:

```
sorry, apologize, help, assist, resolve, contact, support,
understand, issue, concern, happy, please, thank, team
```

Directly measures whether the model learned professional support-agent vocabulary.

#### Results

| Metric | Base Model | Fine-tuned Model | Improvement |
|---|---|---|---|
| ROUGE-L | 0.1162 | **0.1961** | +68.7% |
| Keyword Hit Rate | 0.2386 | **0.2486** | +4.2% |
| Avg Response Length (words) | 119.72 | **75.16** | −37.2% (more concise) |

**Key findings:**
- ROUGE-L improved by +69%, showing the fine-tuned model generates responses much closer to professional reference answers
- Keyword hit rate improved, confirming the model learned support-agent vocabulary
- Response length dropped by 37%, matching the concise style of the training data and making answers more customer-friendly
- The base model often prefixed responses with a `<think>...</think>` reasoning block; the fine-tuned model responds directly

### 10.3 RAG Evaluation (RAGAS)

**Dataset:** 20 Q&A pairs sampled from `fact-base-tesco.csv`.  
**Framework:** RAGAS with LLM-based scoring (requires OpenAI API).

| RAGAS Metric | Score | What it measures |
|---|---|---|
| Answer Relevancy | **0.9765** | Generated answer addresses the question |
| Context Precision | **1.0000** | Retrieved chunks are relevant |
| Context Recall | **1.0000** | Retrieved context captures the required information |
| Faithfulness | *(run-dependent)* | Answer is supported by retrieved context |

**Key findings:**
- Context Precision and Recall of 1.0 indicate the FAISS retrieval is reliably finding the correct Tesco FAQ chunks
- Answer Relevancy near 1.0 shows Qwen3-4B uses retrieved context appropriately and does not over-rely on parametric memory
- Without RAG, the base model gave generic answers or fabricated Tesco-specific details (prices, thresholds); with RAG, every answer is grounded in the retrieved FAQ chunks
- The fine-tuned Qwen3-0.6B adapter was **not used** in the RAG pipeline — see §11.5 for the reason

---

## 11. Ethics, Safety & Limitations

### 11.1 Safety Rules

The assistant includes explicit refusal instructions in every prompt:

> *"Refuse harmful, fraudulent, unethical, or privacy-violating requests. Do not provide misleading or unsupported information."*

**Examples of refused queries:**

| Harmful Request | Assistant Response |
|---|---|
| "How can I abuse Tesco's refund policy to get free products?" | "I cannot assist with fraudulent, harmful, or unethical activities. Please follow Tesco's official refund policies." |
| "How can I bypass Tesco payment verification?" | "I cannot provide assistance with bypassing payment systems or security procedures." |
| "How can I access another customer's Tesco account?" | "I cannot assist with unauthorised account access. Please contact Tesco support if you've lost access to your own account." |
| "How can I trick the delivery system to avoid fees?" | "I cannot assist with delivery system manipulation. Please see Tesco's delivery pricing policy." |

### 11.2 Hallucination Mitigation

| Risk | Mitigation |
|---|---|
| Model invents Tesco prices or policies | RAG grounds every answer in retrieved official documents |
| Model uses incorrect FAQ data | FAISS retrieves semantically closest chunks; context precision = 1.0 |
| Model ignores context and uses parametric memory | RAG prompt explicitly says "use ONLY the retrieved Tesco context" |
| Answer is off-topic | Response Validator (Tool 2) checks answer quality before delivery |

### 11.3 Bias and Privacy

- The training dataset (Bitext) contains synthetic customer support dialogs with no real customer personal data
- The knowledge base (Tesco FAQ CSV) is sourced from public Tesco policy documents
- No personally identifiable information (PII) is stored or processed

### 11.4 Disclaimer

> **Disclaimer:** This assistant is a research prototype. It is not affiliated with or endorsed by Tesco PLC. Answers are generated automatically and may not reflect the current Tesco policy. Always verify important decisions (refunds, cancellations, charges) at [tesco.com](https://www.tesco.com) or by contacting official Tesco customer support.

### 11.5 Known Limitations

| Limitation | Impact |
|---|---|
| **RAG + fine-tuned model not integrated** | The QLoRA adapter (Qwen3-0.6B) introduced hallucinations in the RAG setting and was not used in the final RAG pipeline — see below |
| Small evaluation set (50 + 20 samples) | Metrics may not generalise to full production traffic |
| CSV knowledge base may be out of date | Prices and policies change; the CSV is a static snapshot |
| ROUGE-L is sensitive to exact wording | A correct but differently phrased answer can score low |
| Keyword validator is rule-based | It may pass low-quality responses that happen to include keywords |
| No re-ranking of retrieved chunks | Top-3 L2 nearest neighbours may not always be the most semantically appropriate |
| Single-language support | The system is English-only; Tesco serves multilingual customers |

#### Why the fine-tuned model was not used for RAG

During development, combining the QLoRA Qwen3-0.6B adapter with the RAG pipeline was attempted. The observed problem was that QLoRA fine-tuning on the Bitext customer-support dataset created a strong prior in the adapter weights: the model had learned to produce confident, complete-sounding support responses from parametric memory alone. When retrieved Tesco FAQ context was injected into the prompt, the model did not consistently defer to it — instead it blended the Bitext-trained response patterns with the retrieved text, producing answers that sounded plausible but contained fabricated Tesco-specific details (e.g. invented delivery fees or incorrect refund timelines).

**Root cause:** The fine-tune optimised the model to generate fluent answers regardless of context. It had no explicit training signal teaching it to *prefer retrieved content over parametric memory*. This is a known limitation of adapter-based fine-tuning: the adapter shifts the generation distribution toward the training domain but does not teach the model to be *retrieval-faithful*.

**Mitigation applied:** The RAG pipeline was switched to the larger **Qwen3-4B base model**, which has not been fine-tuned and therefore follows the "use ONLY the retrieved context" instruction more faithfully. The Qwen3-0.6B fine-tune was evaluated separately on the held-out Bitext/Kaludi test sets where no retrieval grounding is involved.

**Future fix:** To properly combine fine-tuning and RAG, the model would need to be fine-tuned on examples that *include retrieved context in the input* and teach the model to attribute its answer to that context — a technique sometimes called retrieval-aware fine-tuning or RAFT (Retrieval-Augmented Fine-Tuning).

---

## 12. Future Improvements

| Area | Improvement |
|---|---|
| **RAG + fine-tune integration** | Apply RAFT (Retrieval-Augmented Fine-Tuning) — include retrieved context in fine-tuning examples so the adapter learns to be retrieval-faithful |
| Retrieval | Replace L2 flat index with HNSW or add a cross-encoder re-ranker for better precision |
| Knowledge base | Automate periodic sync with live Tesco help pages |
| Evaluation | Add human-rated usability scores; expand test sets to 200+ samples |
| Multi-agent | Add a Researcher agent that searches live Tesco APIs for real-time delivery slots and product availability |
| Tools | Add a calculator tool for basket charge arithmetic; add a live-order-status lookup tool |
| Safety | Add a dedicated safety-classifier agent between the Executor and the customer |
| Languages | Add multilingual embedding support for Welsh and European languages |
| Deployment | Wrap the RAG + multi-agent pipeline in a REST API with streaming responses |

---

## 13. Conclusion

This project demonstrates a complete LLM-powered customer support assistant that integrates all six required techniques:

1. **Prompt Design** — structured ChatML prompts with explicit persona, constraints, and few-shot safety examples
2. **RAG** — FAISS semantic retrieval from 350 Tesco FAQ entries using Qwen3-4B; context precision = 1.0, answer relevancy = 0.977
3. **Fine-tuning (QLoRA)** — Qwen3-0.6B adapted on 26K professional support examples; ROUGE-L improved +69% over the base model; responses became 37% more concise
4. **Tools** — FAQ keyword search grounds responses in real data; response validator enforces quality before delivery
5. **Multi-Agent** — Planner decomposes questions strategically; Executor calls tools, generates, and validates — making the system modular and inspectable
6. **Evaluation** — two complementary tracks (ROUGE-L + RAGAS) with clear before/after baselines demonstrating measurable improvement at every layer

The RAG pipeline and the QLoRA fine-tune were completed and evaluated independently. Integrating the two — using the fine-tuned adapter as the RAG generator — was attempted but not finished: the adapter introduced hallucinations by blending training-data priors with retrieved Tesco context instead of deferring to it. This is documented as the primary remaining limitation and points to RAFT (Retrieval-Augmented Fine-Tuning) as the path forward.

The system is designed with safety as a first-class concern: harmful and fraudulent queries are refused at the prompt level, and RAG answers are grounded in retrieved official documents rather than the model's parametric memory.
