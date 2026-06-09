"""
SIMPLE CUSTOMER SUPPORT BOT - Step by Step
===========================================

This is a SIMPLE version that shows exactly what's happening.
No complex architecture - just plain Python showing each step.

STEPS:
1. Load the Qwen model
2. Load FAQ data
3. Planner analyzes the question
4. Executor searches FAQ (TOOL 1)
5. Executor generates response using FAQ results
6. Executor validates response (TOOL 2)
7. Return final response
"""

# ============================================================================
# GLOBAL VARIABLES - Store loaded model to avoid reloading
# ============================================================================

model = None
tokenizer = None
device = None


# ============================================================================
# STEP 1: LOAD THE MODEL (Qwen3-4B)
# ============================================================================



def load_model():
    """
    Load Qwen3-4B model only once.
    If already loaded, return the existing model.
    """
    global model, tokenizer, device
    
    # Check if model is already loaded
    if model is not None and tokenizer is not None:
        print("\n✓ Model already loaded (using cached version)")
        return model, tokenizer, device
    
    print("\n" + "="*70)
    print("STEP 1: Loading Qwen3-4B Model")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        MODEL_NAME = "Qwen/Qwen3-4B"
        print(f"Loading {MODEL_NAME}...")
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model without device_map (simpler approach)
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        # Move to device
        model = model.to(device)
        print(f"✓ Model loaded on {device}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTo fix, try installing missing dependencies:")
        print("  pip install accelerate")
        print("  pip install transformers torch")
        print("\nNote: First run downloads ~7GB. If error, check internet connection.")
        model = None
        tokenizer = None
        device = None
        return None, None, None


# ============================================================================
# STEP 2: LOAD FAQ DATA
# ============================================================================
import pandas as pd
print("\n" + "="*70)
print("STEP 2: Loading FAQ Database")
print("="*70)
faq_df = pd.read_csv('fact-base-tesco.csv')

# Simple FAQ database (like Tesco)
faq_database = [
    {"q": row['Question'], "a": row['Answer']} 
    for _, row in faq_df.iterrows()
]
print(f"✓ Loaded {len(faq_database)} FAQ items")
print("\nFAQ Database contains:")
for i, item in enumerate(faq_database, 1):
    print(f"  {i}. {item['q']}")


# ============================================================================
# STEP 3: TOOL 1 - FAQ SEARCH TOOL
# ============================================================================

print("\n" + "="*70)
print("STEP 3: Define FAQ Search Tool")
print("="*70)

def search_faq_tool(customer_question: str, top_k: int = 3) -> str:
    """
    TOOL 1: Search the FAQ database
    
    This is the first tool that agents can use.
    It searches the FAQ for relevant entries.
    """
    print(f"\n  [TOOL 1 CALLED] Searching FAQ for: '{customer_question}'")
    
    # Simple keyword matching
    results = []
    question_words = customer_question.lower().split()
    
    for faq_item in faq_database:
        score = 0
        for word in question_words:
            if word in faq_item['q'].lower():
                score += 2
            if word in faq_item['a'].lower():
                score += 1
        
        if score > 0:
            results.append((score, faq_item))
    
    # Sort and return top K
    results.sort(reverse=True)
    top_results = results[:top_k]
    
    # Format output
    output = "FAQ Search Results:\n"
    if not top_results:
        output = "No relevant FAQ found."
    else:
        for i, (score, item) in enumerate(top_results, 1):
            output += f"\n{i}. Q: {item['q']}\n   A: {item['a']}\n"
    
    print(f"  [TOOL 1 RESULT] Found {len(top_results)} matches")
    return output


# ============================================================================
# STEP 4: TOOL 2 - RESPONSE VALIDATOR TOOL
# ============================================================================

print("\n" + "="*70)
print("STEP 4: Define Response Validator Tool")
print("="*70)

def validate_response_tool(response: str, question: str) -> str:
    """
    TOOL 2: Validate response quality
    
    This is the second tool. It checks if the response is good.
    """
    print(f"\n  [TOOL 2 CALLED] Validating response quality")
    
    # Simple quality checks
    validation = "Response Validation:\n"
    
    # Check 1: Is it long enough?
    if len(response) > 50:
        validation += "✓ Length: Good (substantial answer)\n"
    else:
        validation += "⚠ Length: Too short\n"
    
    # Check 2: Is it helpful?
    helpful_words = ['yes', 'no', 'can', 'will', 'offer', 'provide']
    if any(word in response.lower() for word in helpful_words):
        validation += "✓ Helpfulness: Clear and direct\n"
    else:
        validation += "⚠ Helpfulness: Could be clearer\n"
    
    # Check 3: Is it professional?
    professional_words = ['thank', 'please', 'help', 'welcome']
    if any(word in response.lower() for word in professional_words):
        validation += "✓ Tone: Professional\n"
    else:
        validation += "⚠ Tone: Could be friendlier\n"
    
    # Quality score
    score = min((len(response) / 100) * 100, 100)
    validation += f"\nQuality Score: {score:.0f}/100"
    
    print(f"  [TOOL 2 RESULT] Score: {score:.0f}/100")
    return validation


# ============================================================================
# STEP 5: AGENT 1 - PLANNER AGENT
# ============================================================================

print("\n" + "="*70)
print("STEP 5: Define Planner Agent")
print("="*70)

def planner_agent(customer_question: str) -> str:
    """
    AGENT 1: PLANNER
    
    This agent analyzes the question and creates a plan.
    """
    print(f"\n{'─'*70}")
    print(f"[PLANNER AGENT] Analyzing question...")
    print(f"{'─'*70}")
    
    # Simple planning logic
    if model is not None and tokenizer is not None:
        prompt = f"""You are a customer support planner. Analyze this question and create a simple 2-step action plan:

Question: {customer_question}

Create a brief 2-step plan to answer this question:
Plan:"""
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            plan = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        except Exception as e:
            plan = f"Plan: 1. Search FAQ for relevant info\n2. Generate helpful response"
    else:
        # Fallback if model not available
        plan = "Plan: 1. Search FAQ for relevant info\n2. Generate helpful response"
    
    print(f"\n[PLANNER OUTPUT]:\n{plan}\n")
    return plan


# ============================================================================
# STEP 6: AGENT 2 - EXECUTOR AGENT (USES TOOLS)
# ============================================================================

print("\n" + "="*70)
print("STEP 6: Define Executor Agent")
print("="*70)

def executor_agent(customer_question: str, planner_plan: str) -> str:
    """
    AGENT 2: EXECUTOR
    
    This agent executes the plan by:
    1. Calling TOOL 1 (FAQ Search)
    2. Generating response
    3. Calling TOOL 2 (Validate)
    """
    print(f"\n{'─'*70}")
    print(f"[EXECUTOR AGENT] Executing plan...")
    print(f"{'─'*70}")
    
    # STEP A: Call TOOL 1 (FAQ Search)
    print("\n[EXECUTOR STEP 1] Call FAQ Search Tool")
    faq_results = search_faq_tool(customer_question)
    print(faq_results)
    
    # STEP B: Generate response using FAQ results
    print("\n[EXECUTOR STEP 2] Generate response using FAQ data")
    
    if model is not None and tokenizer is not None:
        prompt = f"""You are a helpful customer support agent. Use the FAQ information to answer:

Customer Question: {customer_question}

Relevant FAQ Information:
{faq_results}

Provide a helpful, friendly response:
Response:"""
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        except Exception as e:
            response = "I apologize, but I'm having trouble processing your request at the moment."
    else:
        # Fallback simple response
        response = f"Thank you for your question about '{customer_question}'. Based on our FAQ, I can help with that."
    
    print(f"\n[EXECUTOR RESPONSE]:\n{response}\n")
    
    # STEP C: Call TOOL 2 (Validate)
    print("\n[EXECUTOR STEP 3] Validate response quality using Validator Tool")
    validation = validate_response_tool(response, customer_question)
    print(f"\n{validation}\n")
    
    return response


# ============================================================================
# STEP 7: ORCHESTRATOR - PUTS IT ALL TOGETHER
# ============================================================================

print("\n" + "="*70)
print("STEP 7: Orchestrator - Coordinates Agent Collaboration")
print("="*70)

def run_customer_support_bot(customer_question: str):
    """
    ORCHESTRATOR: Manages the entire workflow
    
    Steps:
    1. Planner analyzes question
    2. Executor uses tools to generate response
    3. Return final answer
    """
    print("\n" + "█"*70)
    print("█ CUSTOMER SUPPORT BOT - Multi-Agent System in Action")
    print("█"*70)
    print(f"\nCustomer Question: {customer_question}\n")
    
    # PHASE 1: PLANNING
    print("\n>>> PHASE 1: PLANNING <<<")
    plan = planner_agent(customer_question)
    
    # PHASE 2: EXECUTION (with tools)
    print("\n>>> PHASE 2: EXECUTION (Using Tools) <<<")
    response = executor_agent(customer_question, plan)
    
    # FINAL OUTPUT
    print("\n" + "█"*70)
    print("█ FINAL RESPONSE TO CUSTOMER")
    print("█"*70)
    print(f"\n{response}")
    print("\n" + "█"*70 + "\n")



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function - Executes the customer support bot
    """
    
    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "SIMPLE CUSTOMER SUPPORT BOT" + " "*26 + "║")
    print("║" + " "*10 + "Multi-Agent System with Tool Calling" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    
    # STEP 1: Load model (only loads once, uses cached version if already loaded)
    global model, tokenizer, device
    model, tokenizer, device = load_model()
    
    if model is None:
        print("\n✗ Failed to load model. Exiting.")
        return
    
    print("\n" + "="*70)
    print("STEP 2: Loading FAQ Database")
    print("="*70)
    print(f"✓ Loaded {len(faq_database)} FAQ items")
    print("\nFAQ Database contains:")
    for i, item in enumerate(faq_database, 1):
        print(f"  {i}. {item['q']}")
    
    # Test queries
    test_questions = [
        "Do you offer online shopping?",
        "What's your returns policy?",
        "How can I contact support?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST QUERY {i}/{len(test_questions)}")
        print(f"{'='*70}")
        
        try:
            run_customer_support_bot(question)
        except Exception as e:
            print(f"Error: {e}")
        
        # Only run first query to save time on demo
        if i >= 1:
            print("\n[INFO] Showing first query. Modify to run more queries.")
            break
    
    
if __name__ == "__main__":
    main()
