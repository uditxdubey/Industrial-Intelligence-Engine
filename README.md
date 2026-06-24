# Industrial Intelligence Engine

> Zero-hallucination multi-agent RAG for industrial automation manuals. Siemens vs Rockwell — without data bleeding.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![LLM](https://img.shields.io/badge/LLM-Llama%203.3%2070B-orange)
![Embeddings](https://img.shields.io/badge/embeddings-BAAI%2Fbge--small--en-green)
![VectorDB](https://img.shields.io/badge/vectordb-ChromaDB-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

## The Problem

In industrial automation, a generic RAG system querying multiple vendor manuals produces dangerous hallucinations — mixing Siemens voltage specs with Rockwell terminal configurations. A single wrong answer can cause equipment failure.

Standard flat RAG has no concept of vendor isolation. It retrieves from everything and lets the LLM blend it. That is unacceptable in a production industrial environment.

## The Solution

A **Semantic Multi-Agent Router** that never lets vendor data cross-contaminate.

Instead of one vector database, each vendor gets an isolated store. An LLM router analyzes the user's intent and dispatches to the correct sub-agent — or both, when a comparison is explicitly requested.

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  Agentic Router                 │
│  Groq Llama 3.3 70B             │
│  Classifies intent → routes     │
└──────────┬──────────────┬───────┘
           │              │
     SIEMENS             ROCKWELL
           │              │
           ▼              ▼
┌──────────────┐  ┌──────────────┐
│ Siemens      │  │ Rockwell     │
│ Vector Store │  │ Vector Store │
│ (ChromaDB)   │  │ (ChromaDB)   │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
     LLM Synthesis + Reasoning Trace
                │
                ▼
        Streamlit UI Answer
```

## Architecture Decisions

### Hybrid Ingestion Pipeline
Two data sources, two ingestion strategies — chosen for cost and quality:

| Source | Format | Ingestion Method | Reason |
|--------|--------|-----------------|--------|
| Siemens | Complex PDFs with tables | LlamaParse (Cloud) | Preserves tabular structure critical for specs |
| Rockwell | Text specifications | Local parsing | Zero API cost, sufficient for plain text |

### Standardized Vectorization
Both pipelines converge on the same local embedding model — `BAAI/bge-small-en-v1.5` via HuggingFace. This eliminates embedding dimension mismatches and removes OpenAI embedding costs entirely.

### Explainable Routing
Every answer surfaces a JSON Reasoning Trace in the UI — showing exactly which database was queried, why, and what was retrieved. Engineers can verify the source before acting on the output.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM / Router | Groq — Llama 3.3 70B Versatile |
| PDF Parsing | LlamaParse (Cloud API) |
| Embeddings | BAAI/bge-small-en-v1.5 (local, HuggingFace) |
| Vector Store | ChromaDB (local, isolated collections) |
| UI | Streamlit |
| Language | Python |

## Quickstart

```bash
# 1. Clone
git clone https://github.com/uditxdubey/Industrial-Intelligence-Engine.git
cd Industrial-Intelligence-Engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Add your keys:
# GROQ_API_KEY=your_key
# LLAMA_CLOUD_API_KEY=your_key

# 5. Ingest documents
python ingest_split.py

# 6. Run
streamlit run app.py
```

## Example Queries

```
"What is the maximum input voltage for Siemens S7-1500?"
→ Routes to Siemens agent only

"Compare Rockwell and Siemens safety relay wiring requirements"
→ Routes to both agents, synthesizes comparison

"What are the Rockwell terminal torque specifications?"
→ Routes to Rockwell agent only
```

## Why This Matters

Generic LLM assistants cannot be trusted in industrial environments where wrong answers have physical consequences. This system enforces vendor isolation at the data layer — not at the prompt layer — making hallucination structurally impossible across vendor boundaries.

## Project Structure

```
Industrial-Intelligence-Engine/
├── app.py                  # Streamlit UI + agent orchestration
├── ingest_split.py         # Hybrid ingestion pipeline
├── src/                    # Agent logic and router
│   └── ...
├── data/
│   └── raw/                # Source manuals (PDFs + text)
└── requirements.txt
```

## Built By

Udit Naresh Dubey — Master's in Data Science, FAU Erlangen-Nürnberg  
[LinkedIn](https://www.linkedin.com/in/udit-dubey-9aa0b9284/) · [GitHub](https://github.com/uditxdubey)
