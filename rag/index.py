"""Build or load a persistent Chroma vector store from the cleaned knowledge base."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _flatten_metadata(record: dict) -> dict:
    # Chroma rejects list/dict metadata values — keep only scalar fields.
    keep = ("url", "page_title", "section_title", "content_type", "record_id")
    return {k: record[k] for k in keep if k in record and record[k] is not None}


def load_kb_documents() -> list[Document]:
    if not config.KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"knowledge base not found at {config.KNOWLEDGE_BASE_PATH}; "
            "run run_pipeline.py first"
        )
    records = json.loads(config.KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    return [
        Document(page_content=r["content"], metadata=_flatten_metadata(r))
        for r in records
    ]


def _embeddings() -> OpenAIEmbeddings:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, openai_api_key=api_key)


def _chroma_dir_has_data() -> bool:
    p = Path(config.CHROMA_DIR)
    return p.exists() and any(p.iterdir())


def build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    embedding = _embeddings()

    if not force_rebuild and _chroma_dir_has_data():
        return Chroma(
            persist_directory=config.CHROMA_DIR,
            embedding_function=embedding,
            collection_name=config.COLLECTION_NAME,
        )

    docs = load_kb_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
    )
    return vectorstore


if __name__ == "__main__":
    vs = build_or_load_vectorstore(force_rebuild=True)
    print(f"Vector store contains {vs._collection.count()} vectors")
    print(f"Persisted to {config.CHROMA_DIR}")
