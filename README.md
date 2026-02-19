-- Industrial Agentic RAG: Zero-Hallucination Router-Based Multi-Agent System--

Descrption : An intelligent, multi-agent Retrieval-Augmented Generation (RAG) system designed to query and compare industrial automation manuals (Siemens vs. Rockwell) without data bleeding or hallucination.

The Evolution: From Single-Agent to Multi-Agent Router
This project started as a standard flat RAG querying a single Siemens PDF. However, in industrial automation, asking a generic AI to compare two different brands often leads to dangerous hallucinations (e.g., mixing up voltage terminals).

The Solution: I migrated the architecture to a Semantic Multi-Agent Router.
Instead of one massive database, the system uses an LLM to analyze the user's intent and route the query to isolated, brand-specific Vector Databases. 



  System Architecture & Flow
1.User Query: Entered via Streamlit UI.
2. Agentic Router (Groq Llama 3.3 70B): Analyzes the prompt and selects the appropriate sub-agent(s) based on tool descriptions.
3. Hybrid Ingestion & Retrieval:
 -Siemens Agent: Queries a vector store built from complex PDFs parsed via cloud-based LlamaParse.
-Rockwell Agent: Queries a localized vector store built from raw text files to optimize speed and API costs.
4. Synthesis & Transparency: The LLM combines the retrieved facts and displays the final answer alongside a JSON "Reasoning Trace" so engineers can verify the sources.

Challenges & Debugging
Building a multi-agent system locally presented several unique challenges: 
1.Encountered ModuleNotFoundError for libraries like llama-index-llms-ollama or llama-parse, even after a terminal installation.
-Fix: Verified the active Conda environment (agentic_rag) and ensured the VS Code Python Interpreter was explicitly set to the Conda environment path to link the "Brain".

1.Model Deprecation Crashes: Encountered `model_decommissioned` and `model_not_found` errors when Groq retired the Llama 3 70B `8192` tag. 
-Fix: Debugged API endpoints and migrated the routing engine to the updated `llama-3.3-70b-versatile` standard.
2.Vector Dimension Mismatches: Mixing OpenAI embeddings for cloud parsing with HuggingFace for local parsing corrupted the comparison logic. 
-Fix: Forced global initialization of the local `BAAI/bge-small-en-v1.5` embedding model across all ingestion and retrieval scripts.
3.Cost & Latency Management:** Parsing large technical PDFs drained cloud API limits quickly. 
-Fix: Implemented a conditional routing pipeline in `ingest_split.py` to handle heavy PDFs via Cloud and lightweight competitor text files via local zero-cost processing.

  --How to Run Locally--
1. Clone the repository:
git clone [https://github.com/uditxdubey/Industrial-Agentic-RAG.git](https://github.com/uditxdubey/Industrial-Agentic-RAG.git)
cd Industrial-Agentic-RAG

2. Create a virtual environment (Recommended):
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Set up the environment variables
Create a .env file in the root directory and add your API keys:
GROQ_API_KEY=#enter your personal key here for groq
LLAMA_CLOUD_API_KEY=#enter your key here for LLama Cloud

5. Run the Application
streamlit run app.py

My Notes:
The Goal: Build a "Zero-Hallucination" AI assistant for industrial automation engineers to query and compare complex hardware manuals (Siemens vs. Rockwell) without cross-brand data contamination.

The Local-First Attempt (Privacy Focus): Initially deployed the system entirely locally using Ollama (Llama 3) on an M2 MacBook to guarantee complete data privacy. While secure, inference latency was too high for a production-grade user experience.

The Hybrid Architecture Pivot: Shifted to a split computing model. Offloaded the heavy LLM reasoning to the cloud using Groq (Llama-3.3-70b-versatile) for near-instant inference, while maintaining control over the data layer.

Cost-Optimized Data Ingestion: Engineered a dual-path ingestion pipeline. Complex Siemens PDFs were routed through LlamaParse (Cloud API) to preserve tabular data, while simple Rockwell text specs were parsed locally to conserve API credits.

Standardized Local Vectorization: Forced both data streams through a local HuggingFace embedding model (BAAI/bge-small-en-v1.5). This eliminated embedding costs and standardized the vectors before saving them to isolated collections in a local ChromaDB instance.

The Agentic Router Upgrade: Replaced the flat RAG architecture with an intelligent Semantic Router. The Groq LLM now acts as a manager, analyzing the user's intent and dynamically dispatching queries to the isolated Siemens and/or Rockwell vector stores.

Explainable UI Verification: Wrapped the backend in a Streamlit interface featuring a "Reasoning Trace" expander. This exposes the router's JSON decision-making process, proving to the user exactly which database was queried and why.

