"""Multi-query + RAG-fusion retrieval chain and answering chain.

Expects OPENAI_API_KEY in the environment (load via .env + python-dotenv).
"""
from __future__ import annotations

import json
import sys
from operator import itemgetter
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

import config
from rag.index import build_or_load_vectorstore


_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """Generate {n} diverse search queries for retrieving documents that answer the user's question.
Include paraphrases, synonyms, and related terminology that may appear on a webpage.

Question: {question}

Output one query per line."""
)

_CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation so far and a follow-up question, rewrite the follow-up as a standalone question that can be understood without the prior turns. Resolve pronouns and implicit references using the conversation. If the follow-up is already standalone, return it unchanged. Output only the rewritten question, with no preamble.

Conversation:
{chat_history}

Follow-up: {question}

Standalone question:"""
)

_ANSWER_SYSTEM = """You are an assistant for the University of Chicago Master of Science in Applied Data Science (MS-ADS) program. Your job is to help prospective and current students with accurate information drawn from the program's official materials.

Rules:
1. Answer ONLY using the numbered sources provided below. Do not use outside knowledge.
2. If the sources do not contain the answer, reply briefly that you don't have that information in the MS-ADS materials and suggest checking the program website. Do not guess or fabricate details.
3. Cite every factual claim with bracketed source numbers like [1] or [2, 3]. Use the numbers exactly as shown in the Sources block.
4. Be concise and direct. Use short paragraphs or bullet lists when helpful.
5. Do not invent URLs, dates, prices, or names. If a number or fact is not in the sources, say so."""

_ANSWER_HUMAN = """Sources:
{context}

Question: {question}"""

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _ANSWER_SYSTEM),
        ("human", _ANSWER_HUMAN),
    ]
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
    """Single-turn retrieval: question → multi-query → fused docs."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVAL_K},
    )
    generate_queries = build_multi_query_chain(llm)
    return generate_queries | retriever.map() | reciprocal_rank_fusion


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "(no prior turns)"
    turns = history[-2 * config.NUM_CONDENSE_HISTORY_TURNS:]
    lines = []
    for msg in turns:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines) if lines else "(no prior turns)"


def build_condense_chain(llm: ChatOpenAI):
    """{question, chat_history} → standalone question string.

    If chat_history is empty, the prompt instructs the LLM to return the question
    unchanged, so this is effectively a no-op on the first turn.
    """
    prepare = RunnableLambda(
        lambda x: {
            "question": x["question"],
            "chat_history": _format_history(x.get("chat_history")),
        }
    )
    return prepare | _CONDENSE_PROMPT | llm | StrOutputParser() | RunnableLambda(lambda s: s.strip())


def build_history_aware_retrieval_chain(vectorstore: Chroma, llm: ChatOpenAI):
    """{question, chat_history} → fused docs, with history-aware question rewriting.

    Pipeline: condense (history-aware rewrite) → multi-query expansion → per-query
    similarity search → reciprocal rank fusion.
    """
    condense = build_condense_chain(llm)
    multi_query = build_multi_query_chain(llm)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVAL_K},
    )
    # condense outputs a string; multi_query's prompt template needs {"question": ...}.
    to_dict = RunnableLambda(lambda q: {"question": q})
    return condense | to_dict | multi_query | retriever.map() | reciprocal_rank_fusion


def _dedup_docs(docs: list[Document]) -> list[Document]:
    seen: set[tuple[str, str]] = set()
    out: list[Document] = []
    for d in docs:
        url = (d.metadata.get("url") or "").strip()
        section = (d.metadata.get("section_title") or "").strip()
        key = (url, section)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _format_numbered_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        title = (d.metadata.get("page_title") or "Untitled").strip()
        section = (d.metadata.get("section_title") or "").strip()
        header = f"[{i}] {title}"
        if section:
            header += f" — {section}"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n".join(blocks) if blocks else "(no sources retrieved)"


def build_rag_chain(vectorstore: Chroma, llm: ChatOpenAI):
    """Backwards-compatible single-turn RAG chain returning an answer string.

    Kept so existing callers (e.g., direct chain.invoke) continue to work. New code
    should prefer answer_with_history, which also returns the ordered sources.
    """
    retrieval_chain = build_retrieval_chain(vectorstore, llm)

    def _retrieve_and_format(inputs: dict[str, Any]) -> dict[str, Any]:
        docs = retrieval_chain.invoke({"question": inputs["question"]})
        docs = _dedup_docs(docs)[: config.TOP_N_SOURCES]
        return {"context": _format_numbered_context(docs), "question": inputs["question"]}

    return RunnableLambda(_retrieve_and_format) | _ANSWER_PROMPT | llm | StrOutputParser()


def answer_with_history(
    question: str,
    chat_history: list[dict] | None = None,
    vectorstore: Chroma | None = None,
    llm: ChatOpenAI | None = None,
    retrieval_chain=None,
) -> dict:
    """Run the full chain and return both the answer and the ordered, deduped sources.

    The returned `sources` list is in the same order the LLM was asked to cite, so
    the UI can prefix each card with [N] and the numbers will line up with the
    bracketed citations in `answer`.
    """
    llm = llm or _chat_llm()
    if retrieval_chain is None:
        vectorstore = vectorstore or build_or_load_vectorstore()
        retrieval_chain = build_history_aware_retrieval_chain(vectorstore, llm)

    fused = retrieval_chain.invoke({"question": question, "chat_history": chat_history or []})
    sources = _dedup_docs(fused)[: config.TOP_N_SOURCES]
    context = _format_numbered_context(sources)

    answer_chain = _ANSWER_PROMPT | llm | StrOutputParser()
    answer = answer_chain.invoke({"context": context, "question": question})

    return {"answer": answer, "sources": sources}


def answer_with_sources(question: str, top_n_sources: int | None = None) -> dict:
    """CLI-friendly wrapper: same shape as before but uses the new pipeline."""
    result = answer_with_history(question, chat_history=[])
    sources = result["sources"][: top_n_sources or config.TOP_N_SOURCES]

    labels: list[str] = []
    for d in sources:
        url = d.metadata.get("url", "")
        title = d.metadata.get("page_title", "")
        label = f"{title} — {url}" if title and url else (url or title or "unknown")
        if label not in labels:
            labels.append(label)
    return {"answer": result["answer"], "sources": labels}


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
    for i, s in enumerate(result["sources"], start=1):
        print(f" [{i}] {s}")
