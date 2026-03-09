import torch # Imports PyTorch for handling mathematical model operations
from langchain_chroma import Chroma # Database to store and retrieve document vectors
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline # Local AI interfaces
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline # Loads local models
from langchain_core.messages import HumanMessage, SystemMessage # Standard message objects

# --- 1. SETUP EMBEDDINGS (Local) ---
# We replace OpenAIEmbeddings with a local HuggingFace model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 2. SETUP VECTOR STORE ---
persistent_directory = "db/Chroma_db"
# Connecting to your existing local Chroma database
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# --- 3. SETUP LOCAL LLM (The "Brain") ---
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct" # A small, efficient model for local use

# Tokenizer converts your text prompt into numbers for the model
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Loads the model architecture onto your computer (using GPU if available)
model_obj = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)

# Pipeline bundles the tokenizer and model for easy text generation
pipe = pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    max_new_tokens=500, # Max length of the answer
    temperature=0.1, # Makes the output consistent
    do_sample=True
)

# Wraps the pipeline so it acts like a LangChain object
llm = HuggingFacePipeline(pipeline=pipe)

# --- 4. RETRIEVAL ---
query = "How much did Microsoft pay to acquire Github?"
retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# --- 5. PROMPT FORMATTING ---
# Local models need a specific "Instruct" format to distinguish between System and User
context = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])
formatted_prompt = f"""<|im_start|>system
You are a helpful assistant. Use only the provided documents to answer the question. If you don't know the answer, say so.<|im_end|>
<|im_start|>user
Question: {query}
Documents:
{context}
<|im_end|>
<|im_start|>assistant
"""

# --- 6. GENERATION ---
# Invoke the local LLM with the formatted string prompt
result = llm.invoke(formatted_prompt)

# Clean output: Local models often return the prompt + answer, so we strip the prompt part
answer = result.split("<|im_start|>assistant")[-1].strip()

print("\n--- Generated Response ---")
print(answer)