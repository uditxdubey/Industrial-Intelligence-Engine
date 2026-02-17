import os
import torch
import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables (like your LlamaCloud key if needed)
load_dotenv()

def create_memory():
    print("--- Starting Phase 2: Building Vector Memory ---")

    # 1. M2 ACCELERATION LOGIC
    # This uses the M2 GPU (Metal) instead of the CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 M2 GPU (Metal) detected! Using MPS for high-speed indexing.")
    else:
        device = "cpu"
        print("🐢 Using CPU for indexing.")

    # 2. LOCAL EMBEDDING SETTINGS
    # We use a local model and disable OpenAI to avoid "No API Key" errors
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device=device
    )
    Settings.llm = None  # Prevents LlamaIndex from looking for OpenAI during this step

    # 3. DATABASE INITIALIZATION
    # This creates the 'chroma_db' folder in your project root
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("siemens_s71200")

    # 4. VECTOR STORE CONFIGURATION
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. LOADING DATA
    # This reads the 'parsed_manual.md' file you generated earlier
    print("Reading your parsed Siemens manual...")
    if not os.path.exists("./data/parsed_manual.md"):
        print("❌ Error: ./data/parsed_manual.md not found! Run ingest.py first.")
        return

    documents = SimpleDirectoryReader("./data").load_data()

    # 6. CREATING THE INDEX (THE "BRAIN")
    print("Indexing documents... This may take a minute on your M2.")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )

    print("\n✅ SUCCESS: Local Vector Database created in /chroma_db")
    return index

if __name__ == "__main__":
    create_memory()