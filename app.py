import os
import chromadb
import torch
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

load_dotenv()

def start_hybrid_agent():
    print("--- Initializing Fast Hybrid Siemens Agent ---")

    # 1. Keep Embeddings LOCAL (Fast on M2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5", 
        device=device
    )

    # 2. Use GROQ for the LLM (Lightning Fast Cloud)
    # This uses Llama 3 70B - much smarter than the local one!
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

    # 3. Connect to local ChromaDB
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("siemens_s71200")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=StorageContext.from_defaults(vector_store=vector_store)
    )

    query_engine = index.as_query_engine(similarity_top_k=5)

    print("\n🚀 HYBRID AGENT READY. Responses will be instant now.")
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ["exit", "quit"]:
            break
            
        print("Searching local manual and generating answer...")
        response = query_engine.query(user_query)
        print(f"\nAgent: {response}")

if __name__ == "__main__":
    start_hybrid_agent()