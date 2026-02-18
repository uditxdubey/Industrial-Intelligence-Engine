import os
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.vector_store import get_vector_store_context
from src.ingestion import process_and_index  # <--- This handles the Cloud Parsing (PDFs)

# 1. Load Environment Variables
load_dotenv()

# Forcing Local Embeddings (Crucial for cost & speed and to prevent connecting to OPENAI by deault)
# This ensures Siemens and Rockwell use the exact same "language" (vectors).
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def main():
    print("Initiating Agentic (Hybrid) Ingestion Pipeline--")
    base_raw_path = "data/raw"

    # Using Hash map to reduce the time complexity 
    collections = {
        "siemens": "siemens_knowledge_base",
        "competitors": "rockwell_knowledge_base"
    }

    # this loop ensaures that the agents cant see each other data thus not hallucinate
    for category in os.listdir(base_raw_path):
        category_path = os.path.join(base_raw_path, category)
        
        # Skipping hidden files 
        if os.path.isdir(category_path) and not category.startswith("."):
            
            # Determining the Target Collection 
            target_collection = collections.get(category, "general_knowledge_base")
            print(f"\n Processing '{category}' -> Routing to Collection: [{target_collection}]")
            
            # Open the correct "Vault"
            storage_context = get_vector_store_context(collection_name=target_collection)
            
            
            
            # STRATEGY A: High-Fidelity Parsing Siemens;
            # We use LlamaParse (Cloud) because these manuals have complex tables.
            if category == "siemens":
                print(" Strategy: Cloud Parsing (LlamaParse) for Complex PDFs...")
                try:
                    process_and_index(category_path, category, storage_context)
                    print(" Siemens Data Ingested.")
                except Exception as e:
                    print(f"  Skipping Siemens (likely already done or API limit): {e}")

            # STRATEGY B: Fast Local Parsing (Competitors / Text)
            # We use SimpleDirectoryReader (Local) because these are simple text files.
            elif category == "competitors":
                print("Strategy: Local Parsing (SimpleReader) for Text Files...")
                
                documents = SimpleDirectoryReader(category_path).load_data()
                
                # Tag Metadata (Crucial for Citations later)
                for doc in documents:
                    doc.metadata["brand"] = "rockwell"
                    doc.metadata["category"] = "competitor_specs"

                # Embed and Save to Disk
                print(f"Inserting {len(documents)} docs into Vector Store--")
                VectorStoreIndex.from_documents(documents, storage_context=storage_context)
                print(" Rockwell Data Ingested.")

    print("\nINGESTION SUCCESSFUL-- All Agents are ready.")

if __name__ == "__main__":
    main()