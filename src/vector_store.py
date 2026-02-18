import chromadb #local database for AI: stores numbers(vectors) instead of text
from llama_index.vector_stores.chroma import ChromaVectorStore #bridge that allows LlamaIndex to communicate with ChromaDB
from llama_index.core import VectorStoreIndex, StorageContext#suitcase that holds the vector stores

def get_vector_store_context(collection_name, path="./chroma_db"): #used by ingest_split.py . it prepares the database to receive new data
    
    db = chromadb.PersistentClient(path=path)#connect to the local ChromaDB instance. If it doesn't exist, it will create a new one at the specified path.
    
    # get_or_create ensures we can add to it if it exists, or make a new one,preventing data bleed so the agents cant see each other as they are in different collection
    chroma_collection = db.get_or_create_collection(collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)#formatting the chroma collection for LLamaIndex to understand. It acts as a translator between the two systems.
    storage_context = StorageContext.from_defaults(vector_store=vector_store)#suitcase
    
    return storage_context

def load_index_from_disk(collection_name, path="./chroma_db"):#used by app.py to load the specific index for router
    
    db = chromadb.PersistentClient(path=path)
    chroma_collection = db.get_collection(collection_name)#using get; makes sure that the ai crashes before using a ghost query and hallucinate
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    return VectorStoreIndex.from_vector_store(vector_store)#rerouting the index