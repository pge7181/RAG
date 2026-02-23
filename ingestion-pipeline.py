import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all the text files from the repository"""
    print(f"Loading documents from {docs_path}...")

    # Check if the directory exists

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\n Document: {i+1}")
        print(f"   Source: {doc.metadata['source']}")
        print(f"   Content length: {len(doc.page_content)} characters")
        print(f"   Content preview: {doc.page_content[:100]}...")
        print(f"    metadata: {doc.metadata}")
    
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Splitting documents into smaller chunks overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)

        if len(chunks) > 5:
            print(f"\n.. and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store using FREE local embeddings"""
    print("Creating local embeddings and storing in ChromaDB...")

    # This model is free, small, and runs on your CPU/Laptop
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    # Create ChromaDB vector store
    print("--- Creating vector store (Local) ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"--- Finished! Vector store saved to {persist_directory} ---")
    return vectorstore



def main():
    print("Main Function")

    # 1. Loading the Files
    documents = load_documents(docs_path="docs")
    # Chunking the files
    chunks = split_documents(documents)
    # 3. Embedding and storing in vector db
    vectorstore = create_vector_store(chunks)



if __name__ == "__main__":
    main()