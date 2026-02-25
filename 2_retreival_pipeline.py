import os
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

# --- 1. SETUP MODEL (Do this ONCE outside the loop) ---
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model_obj = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

pipe = pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.1,
    do_sample=True,
    model_kwargs={"attn_implementation": "eager"} 
)

model = HuggingFacePipeline(pipeline=pipe)

# --- 2. SETUP RETRIEVER ---
persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)
retriever = db.as_retriever(search_kwargs={"k": 5})

queries = [
    "What was Microsoft's first hardware product release?",
    "In what year did Tesla begin production of the Roadster?"
]

# --- 3. RUN PIPELINE ---
for query in queries:
    # Get relevant docs
    relevant_docs = retriever.invoke(query)
    context = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])
    
    # Format prompt using the specific SmolLM2 tags
    formatted_prompt = f"""<|im_start|>system
You are a helpful assistant who answers questions based ONLY on provided documents.<|im_end|>
<|im_start|>user
Question: {query}

Documents:
{context}

Answer:<|im_end|>
<|im_start|>assistant
"""

    # Generate response
    raw_result = model.invoke(formatted_prompt)

    # Clean the output (isolate the assistant's part)
    if "<|im_start|>assistant" in raw_result:
        answer = raw_result.split("<|im_start|>assistant")[-1].strip()
    else:
        answer = raw_result.strip()

    print(f"\nUser Query: {query}")
    print("-------------Generated Response-----------------")
    print(answer)
    print("-" * 50)