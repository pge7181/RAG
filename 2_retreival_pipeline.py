from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"

# Loading Embeddings and Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# search for relevant documents
# query="Which island does SpaceX lease for its launches in the Pacific?"

queries = [
    "Which island does SpaceX lease for its launches in the Pacific?",
    "What was NVIDIA's first graphics accelerator called?",
    "What was Microsoft's first hardware product release?",
    "How much did microsoft pay to acquire Github?",
    "In what year did Tesla begin production of the Roadster?",
    "Who succeeded Ze'ev Drori as CEO in October 2008",
    "Which island does SapceX lease for its launches in the Pacific?",
    "What was the original name of Microsoft before it became Microsoft?"
]

retriever = db.as_retriever(search_kwargs={"k":5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3 # only return chunks with cosine similarity >=0.3
#     }
# )


for query in queries:
    relevant_docs = retriever.invoke(query)

    print(f"User Query: {query}")
    # Display Results

    print("----Context----")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")