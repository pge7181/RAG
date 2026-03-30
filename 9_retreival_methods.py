from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Swapped for local embeddings
from dotenv import load_dotenv

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"

# --- 1. INITIALIZE LOCAL EMBEDDINGS ---
# This downloads a ~80MB model to your machine once. 
# It performs all vector math locally. No API Key required.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # Change 'cpu' to 'cuda' if you have an NVIDIA GPU
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Query to test
query = "How much did Microsoft pay to acquire GitHub?"
print(f"Query: {query}\n")

# ──────────────────────────────────────────────────────────────────
# METHOD 1: Basic Similarity Search
# ──────────────────────────────────────────────────────────────────

print("=== METHOD 1: Similarity Search (Local k=3) ===")
retriever = db.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents:\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

print("-" * 60)

# ──────────────────────────────────────────────────────────────────
# METHOD 3: Maximum Marginal Relevance (MMR)
# ──────────────────────────────────────────────────────────────────
# MMR is even more important with local models to ensure 
# the retrieved context isn't repetitive.

print("\n=== METHOD 3: Maximum Marginal Relevance (MMR) ===")
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           
        "fetch_k": 10,    
        "lambda_mult": 0.5  
    }
)

docs = retriever.invoke(query)
for i, doc in enumerate(docs, 1):
    print(f"Document {i}: {doc.page_content[:100]}...")

print("=" * 60)