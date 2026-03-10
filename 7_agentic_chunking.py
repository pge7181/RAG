import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# --- 1. SETUP LOCAL LLM ---
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

print("Loading local model for agentic chunking...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_obj = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Pipeline optimized for instruction following
pipe = pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    max_new_tokens=500, # Increased tokens to accommodate the full text processing
    temperature=0.1,    # Keep low for precise, rule-following behavior
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- 2. THE TEXT & PROMPT ---
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Formatted prompt for the Instruct model
prompt = f"""<|im_start|>system
You are a text chunking expert. Split this text into logical chunks.
Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put <<<SPLIT>>> between chunks
<|im_end|>
<|im_start|>user
Text:
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split:
<|im_end|>
<|im_start|>assistant
"""

# --- 3. GENERATION & PARSING ---
print("🤖 Asking local AI to chunk the text...")
result = llm.invoke(prompt)

# Isolate just the assistant's part of the response
marked_text = result.split("<|im_start|>assistant")[-1].strip()

# Split the text at the custom markers
chunks = marked_text.split("<<<SPLIT>>>")

# Clean up whitespace and filter empty chunks
clean_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

# Show results
print("\n🎯 AGENTIC CHUNKING RESULTS (Local):")
print("=" * 50)
for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print("-" * 50)