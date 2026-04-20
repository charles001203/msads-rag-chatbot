# UChicago MS-ADS Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the University of Chicago's Master's in Applied Data Science program. The knowledge base is built by scraping the official program site; retrieval uses multi-query rewriting + Reciprocal Rank Fusion over a persisted Chroma vector store; answers are generated with OpenAI chat models via LangChain.

## Quickstart

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your OpenAI key**
   ```bash
   cp .env.example .env
   # then edit .env and set OPENAI_API_KEY=sk-...
   ```

3. **Build the knowledge base** (one-time; scrapes the MS-ADS site and writes `data/processed/knowledge_base.json`)
   ```bash
   python -m ingest.run_pipeline
   ```

4. **Build the vector index** (one-time; embeds the knowledge base into `data/chroma/`)
   ```bash
   python -m rag.index
   ```

5. **Smoke-test the chain from the CLI**
   ```bash
   python -m rag.chain "What are the career outcomes of a UChicago MS-ADS student?"
   ```

6. **Launch the Streamlit UI**
   ```bash
   streamlit run streamlit_app.py
   ```

## Project layout

```
.
├── config.py              Shared paths, crawl policy, and RAG settings
├── streamlit_app.py       Streamlit UI entrypoint
├── requirements.txt
├── .env.example
│
├── ingest/                Knowledge-base pipeline (scrape → extract → clean)
│   ├── scrape.py
│   ├── extract.py
│   ├── clean.py
│   └── run_pipeline.py    Orchestrator
│
├── rag/                   Retrieval + answering
│   ├── index.py           Builds / loads the persistent Chroma store
│   └── chain.py           Multi-query + RAG-fusion chain and CLI
│
├── data/
│   ├── raw/               Downloaded HTML (gitignored)
│   ├── interim/           Per-section records (gitignored)
│   ├── processed/         knowledge_base.json — the RAG input
│   └── chroma/            Persisted vector store (gitignored)
│
├── logs/                  Pipeline logs
│
└── docs/                  Reference material (assignment brief, original notebook)
```

## How it works

- **Ingest** (`ingest/`): `scrape.py` crawls the MS-ADS site within an allowlist; `extract.py` pulls section-level records with page/section titles and splits FAQ pairs; `clean.py` dedupes boilerplate and near-duplicates. Output is `data/processed/knowledge_base.json`.
- **Index** (`rag/index.py`): loads the knowledge base, chunks long prose with `RecursiveCharacterTextSplitter`, embeds with `text-embedding-3-small`, and persists to Chroma.
- **Retrieve + answer** (`rag/chain.py`): generates N diverse query rewrites, retrieves top-k per query, fuses with Reciprocal Rank Fusion, then answers with `gpt-4o-mini` grounded only on retrieved context.
- **UI** (`streamlit_app.py`): wraps the chain with `@st.cache_resource` so the vector store isn't re-loaded on every rerun; shows the answer with an expandable Sources list.
