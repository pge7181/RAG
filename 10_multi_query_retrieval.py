import torch
import json
import re # Added for robust parsing
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- 1. SETUP ---
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
persistent_directory = "db/chroma_db"

print("Initializing Fast Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

print("Loading Embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

# --- 2. OPTIMIZED MAC LOADING ---
print("Loading Model (bfloat16 optimized)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True # Helps with limited RAM
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150, # Reduced to speed up generation
    temperature=0.1,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- 3. ROBUST EXECUTION ---
original_query = "How does Tesla make money?"
print(f"\nOriginal Query: {original_query}")

# Mistral works better when you give it a clear example
prompt = f"""<s>[INST] Generate 3 search queries for: "{original_query}"
Format your response exactly like this example:
["query 1", "query 2", "query 3"]
Do not provide any intro or outro text. [/INST]"""

print("Generating (Please wait 30-60 seconds)...")
response = llm.invoke(prompt)

# --- 4. ROBUST JSON PARSING ---
try:
    # Use regex to find anything inside square brackets [ ... ]
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        query_variations = json.loads(match.group())
    else:
        raise ValueError("No list found")
except Exception:
    # Reliable fallback if the model fails the format
    query_variations = [original_query, "Tesla revenue sources", "Tesla business model"]

print("\nFinal Query Variations:")
for i, q in enumerate(query_variations, 1):
    print(f"{i}. {q}")

# --- 5. RETRIEVAL ---
retriever = db.as_retriever(search_kwargs={"k": 2}) # Reduced k for speed
for query in query_variations:
    print(f"\n🔍 Searching for: {query}")
    docs = retriever.invoke(query)
    for doc in docs:
        print(f"- {doc.page_content[:100]}...")