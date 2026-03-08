import torch # imports PyTorch, the library used for GPU/CPU math operations
from langchain_chroma import Chroma # Connect to chromaDB to retrieve stored data
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline # Interfaces to run AI tools locally
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline # Loads local AI models and tokenizers

# --- 1. SETUP LOCAL LLM (No API Key Required) ---
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct" # Define the speicific pre-trained AI model we want to use

print("Loading local model...") # Feedback to the user while the model loads into RAM
tokenizer = AutoTokenizer.from_pretrained(model_id) # Loads the translator that turns text into numbers
model_obj = AutoModelForCausalLM.from_pretrained( # Loads the brain, the model itself
    model_id,
    device_map="auto", # Automatically detects if you have GPU/CPU
    torch_dtype=torch.bfloat16 # Uses 16 bit precision to reduce memory usage while keeping performance high
)

pipe = pipeline( # Packages the model into a standard format that performs text generation
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    max_new_tokens=256, # Limites the length of the answer AI provides
    temperature=0.1, # Makes the AI more focussed (less random)
    do_sample=True # Enables variability in text generation
)

# Wrap the pipeline in LangChains interface so it works with the rest of your apps
llm = HuggingFacePipeline(pipeline=pipe)

# --- 2. SETUP RETRIEVER ---
persistent_directory = "db/chroma_db" # The local folder where my vector data is saved 
# Initialize the embedding model that converts your search query into vectors (numbers)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connects to your local chroma DB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

# Creates a retriever object that finds the 'k' most similar documents in the database
retriever = db.as_retriever(search_kwargs={"k": 5})

# defines the sample questions to test the pipeline
queries = [
    "What was Microsoft's first hardware product release?",
    "In what year did Tesla begin production of the Roadster?"
]

# --- 3. RUN PIPELINE ---
for query in queries:
    relevant_docs = retriever.invoke(query)
    
    # Combine query and relevant documents
    context = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])
    
    # Use the specific prompt format for the Instruct model
    # Prompt engineering: Creates a specific template for the 'Instruct' model
    #<|im_start|> is the standard delimeter for this specific model architecture 
    formatted_prompt = f"""<|im_start|>system
You are a helpful assistant who answers questions based ONLY on provided documents.<|im_end|>
<|im_start|>user
Question: {query}

Documents:
{context}

Answer:<|im_end|>
<|im_start|>assistant
"""

    # Generate response by passing the formatted prompt to the local LLM 
    # We use invoke on the llm object directly
    result = llm.invoke(formatted_prompt)

    # Clean the output if the model repeats the prompt, we strip it out to show only the answer
    answer = result.split("<|im_start|>assistant")[-1].strip()

    # Print the result to the console for verification
    print(f"\n--- User Query: {query} ---")
    print(f"Generated Response: {answer}")
    print("-" * 50)