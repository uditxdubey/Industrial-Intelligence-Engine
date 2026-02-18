import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from src.vector_store import get_vector_store_context
from src.retriever import AgenticRouter

# CONFIGURATION ---
st.set_page_config(page_title="Industrial Agentic RAG", page_icon="🏭", layout="wide")
load_dotenv()

# Force Local Embeddings (Must match what we used in Ingestion)
# If we don't do this, it will try to use OpenAI and crash.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Setup Groq LLM 
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error(" GROQ_API_KEY not Found in .env file.")
    st.stop()
    
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)

# LOAD THE AGENTS --
@st.cache_resource(show_spinner="Agents loading-- this happens once per session")
def initialize_system():#using cache to keep it in memory
    #loads the Vector Stores from disk and builds the Router.

    try:
        # Load Siemens Agent
        siemens_ctx = get_vector_store_context("siemens_knowledge_base")
        siemens_index = VectorStoreIndex.from_vector_store(
            vector_store=siemens_ctx.vector_store,
        )

        # Load Rockwell Agent
        rockwell_ctx = get_vector_store_context("rockwell_knowledge_base")
        rockwell_index = VectorStoreIndex.from_vector_store(
            vector_store=rockwell_ctx.vector_store,
        )
        
        # Build the Router
        router = AgenticRouter(llm=Settings.llm)
        return router.create_router_engine(siemens_index, rockwell_index)
        
    except Exception as e:
        st.error(f" Critical Error loading databases: {e}")
        return None

# UI LAYOUT ---
st.title("Router Based Multi-Agent RAG System Designed for Industrial Support ")
st.markdown("""
Architecture: Multi-Agent Router (LlamaIndex + Groq)
Agents: Siemens Specialist (Vector DB) | Rockwell Specialist (Vector DB)
Goal: Zero-Hallucination Industrial Support
""")

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#  THE CHAT LOGIC ---
if prompt := st.chat_input("Ask about wiring, voltage, or compare brands..."):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Response
    with st.chat_message("assistant"):
        query_engine = initialize_system()
        
        if query_engine:
            with st.spinner(" Router is analyzing your intent..."):
                response = query_engine.query(prompt)
            
            # Output the Answer
            st.markdown(response.response)
            
            # Show the "Brain" (Reasoning) - CRITICAL FOR RECRUITERS
            with st.expander(" View Agentic Reasoning (Router Decision)"):
                # Check if we have metadata about the tool selection
                if hasattr(response, 'metadata') and response.metadata:
                    st.json(response.metadata)
                else:
                    st.info("Direct retrieval used.")
                    
                # Show Source Nodes (Citations)
                st.subheader(" Referenced Manuals:")
                for node in response.source_nodes:
                    # Extract metadata safely
                    meta = node.node.metadata
                    score = "{:.2f}".format(node.score) if node.score else "N/A"
                    st.markdown(f"- **{meta.get('file_name', 'Unknown File')}** (Score: {score})")
                    st.caption(f"...{node.node.get_content()[:200]}...")

            # Save context
            st.session_state.messages.append({"role": "assistant", "content": response.response})