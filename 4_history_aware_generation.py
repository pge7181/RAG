import torch # PyTorch handles the mathematical operations on your GPU/CPU
from langchain_chroma import Chroma # The vecotr database to store and retrieve your documents
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline # Interfaces for local AI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline # Loads local LLM architecture
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # LangChain's standrd for chat history

# --- 1. SETUP LOCAL MODEL (No API Key) ---
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct" # Define the specific small language model to load

print("Loading local model... this may take a moment.")
tokenizer = AutoTokenizer.from_pretrained(model_id) # Loads the tokenizer that converts texts to numbers the model understands
model_obj = AutoModelForCausalLM.from_pretrained( # loads the actual model architecture
    model_id, 
    device_map="auto", # automatically assignas model to GPU if available else CPU
    torch_dtype=torch.bfloat16 # Uses 16-bit precision to save RAM and speed up inference
)

pipe = pipeline( # wraps the model into a standard generation pipeline
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    max_new_tokens=256, # limits the length of the AI Answer
    temperature=0.1, # Makes the output deterministic and focussed
    do_sample=True # Allows the slight variation in output
)
local_llm = HuggingFacePipeline(pipeline=pipe) # wraps pipeline so langchain can talk to it

# --- 2. SETUP RETRIEVER ---
persistent_directory = "db/chroma_db" # The folder where my vector data lives.
# Using a local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # converts search queries into vectors
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings) # connect to your local DBs

# Store our conversation as messages
chat_history = [] # Acts as a "short-term memory" for your chatbot

def format_messages_to_prompt(messages):
    """Converts LangChain message list to a ChatML string for local LLM."""
    prompt = ""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif isinstance(msg, HumanMessage):
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif isinstance(msg, AIMessage):
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n" # Tells the model to start generating it's response
    return prompt

# Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    # --- DEBUGGING LINE ---
    print(f"DEBUG: Found {len(docs)} documents.")
    for i, doc in enumerate(docs):
        print(f"DEBUG: Doc {i} Content Preview: {doc.page_content[:100]}")
    # ----------------------

def ask_question(user_question):
    global chat_history # Allows the function to modify the chat_history list defined outside
    
    # --- RESET COMMAND ---
    if user_question.lower() == "reset":
        chat_history.clear() # Wipes memory
        print("Memory cleared!")
        return

    print(f"\n--- You asked: {user_question} ---")
    
    # --- LIMIT MEMORY ---
    if len(chat_history) > 4:
        chat_history = chat_history[-4:] # Prevents the history from getting too big (Sliding window)
    
    # Step 1: Handle standalone question
    search_question = user_question
    if chat_history:
        # We need a temporary prompt to rewrite the question
        standalone_prompt = format_messages_to_prompt(chat_history + [HumanMessage(content=f"Rewrite this as a standalone question: {user_question}")])
        result = local_llm.invoke(standalone_prompt)
        search_question = result.split("<|im_start|>assistant")[-1].strip() # Exytract only the model's text
        print(f"Searching for: {search_question}")
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3}) # Fetches top 3 relevant chunks
    docs = retriever.invoke(search_question)
    
    # --- STEP 3: DEFINITION IS HERE ---
    context = "\n".join([f"- {doc.page_content}" for doc in docs])
    combined_input = f"""Based on the following documents, answer this question: {user_question}

Documents:
{context}

If you can't find the answer in the documents, say "I don't have enough information."
"""
    
    # Step 4: Get the answer
    # Now combined_input is definitely defined
    messages = [SystemMessage(content="You are a helpful assistant.")] + chat_history + [HumanMessage(content=combined_input)]
    formatted_prompt = format_messages_to_prompt(messages)
    
    answer = local_llm.invoke(formatted_prompt)
    
    # Clean up the output
    if "<|im_start|>assistant" in answer:
        answer = answer.split("<|im_start|>assistant")[-1].strip()
    else:
        answer = answer.strip()
    
    # Step 5: Remember conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()