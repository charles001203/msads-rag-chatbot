"""Multi-query + RAG-fusion retrieval chain and answering chain.

Expects OPENAI_API_KEY in the environment (load via .env + python-dotenv).
"""
from __future__ import annotations

import json
import sys
from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config
from rag.index import build_or_load_vectorstore


_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """Generate {n} diverse search queries for retrieving documents that answer the user's question.
Include paraphrases, synonyms, and related terminology that may appear on a webpage.

Question: {question}

Output one query per line."""
)

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the following question based only on this context:

{context}

Question: {question}
"""
)


def _chat_llm() -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0)


def build_multi_query_chain(llm: ChatOpenAI):
    return (
        _MULTI_QUERY_PROMPT.partial(n=str(config.NUM_MULTI_QUERIES))
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )


def reciprocal_rank_fusion(results: list[list[Document]], k: int = 60) -> list[Document]:
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = json.dumps(doc.model_dump(), sort_keys=True)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0.0
                doc_map[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + 1 + k)

    ordered = sorted(doc_map.keys(), key=lambda x: fused_scores[x], reverse=True)
    return [doc_map[d] for d in ordered]


def build_retrieval_chain(vectorstore: Chroma, llm: ChatOpenAI):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVAL_K},
    )
    generate_queries = build_multi_query_chain(llm)
    return generate_queries | retriever.map() | reciprocal_rank_fusion


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_rag_chain(vectorstore: Chroma, llm: ChatOpenAI):
    retrieval_chain = build_retrieval_chain(vectorstore, llm)
    return (
        {
            "context": retrieval_chain | _format_docs,
            "question": itemgetter("question"),
        }
        | _ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )


def answer_with_sources(question: str, top_n_sources: int = 5) -> dict:
    vectorstore = build_or_load_vectorstore()
    llm = _chat_llm()
    retrieval_chain = build_retrieval_chain(vectorstore, llm)
    rag_chain = build_rag_chain(vectorstore, llm)

    retrieved = retrieval_chain.invoke({"question": question})[:top_n_sources]
    answer = rag_chain.invoke({"question": question})

    sources: list[str] = []
    for d in retrieved:
        url = d.metadata.get("url", "")
        title = d.metadata.get("page_title", "")
        label = f"{title} — {url}" if title and url else (url or title or "unknown")
        if label not in sources:
            sources.append(label)
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python -m rag.chain "your question here"', file=sys.stderr)
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    result = answer_with_sources(question)
    print("Question:", question)
    print()
    print("Answer:", result["answer"])
    print()
    print("Sources:")
    for s in result["sources"]:
        print(" -", s)
