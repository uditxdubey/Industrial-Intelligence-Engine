import os
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex

def process_and_index(category_path, category_name, storage_context):
    parser = LlamaParse(result_type="markdown", verbose=True)
    
    for filename in os.listdir(category_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(category_path, filename)
            print(f"☁️ Parsing {filename} in {category_name}...")
            
            # Use LlamaCloud to parse
            documents = parser.load_data(file_path)
            
            # Inject Metadata for Zero-Hallucination
            for doc in documents:
                doc.metadata.update({
                    "brand": category_name,
                    "file_name": filename
                })
            
            # Index to ChromaDB
            VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context,
                show_progress=True
            )