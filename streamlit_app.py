"""Streamlit UI — ChatGPT-style layout for the UChicago MS-ADS chatbot."""
from __future__ import annotations

import html
import uuid

import streamlit as st

from rag.chain import (
    _chat_llm,
    answer_with_history,
    build_history_aware_retrieval_chain,
)
from rag.index import build_or_load_vectorstore


st.set_page_config(
    page_title="UChicago MS-ADS Chatbot",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="expanded",
)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Serif+Pro:wght@400;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --uchi-maroon: #800000;
    --uchi-maroon-dark: #5a0000;
    --uchi-gold: #d6a756;
    --bg: #ffffff;
    --bg-rgb: 255, 255, 255;
    --sidebar-bg: #f7f5f3;
    --border: #e5e0de;
    --border-soft: #efebe9;
    --text: #1f1f1f;
    --muted: #6b6b6b;
    --user-bubble: #f1ebe8;
    --surface: #ffffff;
    --surface-hover: #fcfaf9;
    --surface-alt: #ece5e1;
    --surface-active: #ece0da;
    --expander-bg: #fcfaf9;
    --snippet-text: #333333;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text);
}

[data-testid="stHeader"] { background: transparent; height: 0; }

/* Enlarge the st.logo image */
[data-testid="stLogo"],
img[data-testid="stLogo"],
[data-testid="stSidebarHeader"] img {
    height: 3.0rem !important;
    max-height: 3.0rem !important;
    width: auto !important;
    max-width: 100% !important;
}

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; padding-bottom: 1rem; }

.sidebar-logo {
    padding: 0 0.1rem 0.9rem 0.1rem;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 1rem;
}
.sidebar-logo .product {
    font-family: 'Source Serif Pro', Georgia, serif;
    font-size: 0.88rem;
    color: var(--uchi-maroon);
    font-style: italic;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.sidebar-section-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    font-weight: 600;
    margin: 1rem 0 0.4rem 0.2rem;
}
.sidebar-empty-note {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: var(--muted);
    font-style: italic;
    padding: 0.45rem 0.65rem;
}

/* "New chat" button (primary) — filled maroon */
[data-testid="stSidebar"] button[kind="primary"] {
    background: var(--uchi-maroon) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.55rem !important;
}
[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: var(--uchi-maroon-dark) !important;
}

/* History buttons (secondary) — ghost, left-aligned, truncated */
[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important;
    color: var(--text) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.83rem !important;
    padding: 0.45rem 0.65rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    min-height: auto !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] button[kind="secondary"] > div,
[data-testid="stSidebar"] button[kind="secondary"] p {
    text-align: left !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    width: 100%;
    margin: 0 !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: var(--surface-alt) !important;
    color: var(--uchi-maroon) !important;
}
[data-testid="stSidebar"] button[kind="secondary"][data-active="true"],
[data-testid="stSidebar"] button[kind="tertiary"] {
    background: var(--surface-active) !important;
    color: var(--uchi-maroon) !important;
    font-weight: 500 !important;
}

/* ============ MAIN CHAT AREA ============ */
.block-container {
    max-width: 820px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

/* Empty state */
.empty-state {
    text-align: center;
    margin-top: 10vh;
    margin-bottom: 2rem;
}
.empty-state .emoji {
    font-size: 2.6rem;
    margin-bottom: 0.8rem;
}
.empty-state h1 {
    font-family: 'Playfair Display', Georgia, serif;
    color: var(--uchi-maroon);
    font-weight: 700;
    font-size: 2.1rem;
    margin: 0 0 0.5rem 0;
}
.empty-state p {
    color: var(--muted);
    font-family: 'Source Serif Pro', Georgia, serif;
    font-size: 1rem;
    margin: 0;
}

/* Suggested-prompt cards (secondary buttons in main area) */
.main div[data-testid="stButton"] > button {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: left;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 400;
    line-height: 1.4;
    white-space: normal;
    transition: all 0.15s ease;
    min-height: 3.5rem;
    height: auto;
}
.main div[data-testid="stButton"] > button:hover {
    border-color: var(--uchi-maroon);
    color: var(--uchi-maroon);
    background: var(--surface-hover);
}

/* Chat messages — remove default container chrome */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.45rem 0 !important;
    gap: 0.75rem;
}

/* User turn: right-aligned bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: var(--user-bubble);
    border-radius: 18px 18px 4px 18px;
    padding: 0.7rem 1.1rem;
    max-width: 75%;
    border: 1px solid var(--border-soft);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"] {
    display: none;
}

/* Assistant turn: no bubble, full width */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    padding: 0.3rem 0 0.3rem 0.2rem;
    line-height: 1.6;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    background: var(--uchi-maroon) !important;
    color: white !important;
}

/* Sticky chat input at bottom — strip all wrapper chrome, style only the pill */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] form,
[data-testid="stChatInputContainer"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stBottom"] {
    background: linear-gradient(to top, var(--bg) 75%, rgba(var(--bg-rgb), 0)) !important;
}
[data-testid="stChatInput"] {
    max-width: 820px;
    margin: 0 auto;
    padding: 0.4rem 0 0.6rem 0;
}
[data-testid="stChatInput"] textarea {
    border-radius: 26px !important;
    border: 1px solid var(--border) !important;
    padding: 0.95rem 1.2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--uchi-maroon) !important;
    box-shadow: 0 0 0 2px rgba(128,0,0,0.12) !important;
}

/* Sources expander */
details[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--expander-bg);
    margin-top: 0.6rem;
}
details[data-testid="stExpander"] summary {
    font-family: 'Source Serif Pro', Georgia, serif;
    font-weight: 600;
    color: var(--uchi-maroon);
    font-size: 0.88rem;
}

/* Source cards inside the expander */
.source-card {
    background: var(--surface);
    border-left: 3px solid var(--uchi-maroon);
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    border-radius: 6px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.source-card .title {
    font-family: 'Source Serif Pro', Georgia, serif;
    font-weight: 600;
    color: var(--uchi-maroon);
    font-size: 0.95rem;
    margin-bottom: 0.1rem;
}
.source-card .section {
    color: var(--muted);
    font-size: 0.8rem;
    margin-bottom: 0.4rem;
    font-style: italic;
}
.source-card .snippet {
    color: var(--snippet-text);
    font-size: 0.88rem;
    line-height: 1.55;
    margin-bottom: 0.4rem;
}
.source-card a {
    color: var(--uchi-maroon);
    font-size: 0.8rem;
    text-decoration: none;
    word-break: break-all;
}
.source-card a:hover { text-decoration: underline; }

/* Hide the default "Press Enter to send" helper and Streamlit footer */
[data-testid="stStatusWidget"] { display: none; }
footer { visibility: hidden; }

/* ============ DARK MODE ============
   Triggers:
     1. OS-level dark preference (covers Streamlit's "auto" theme).
     2. Streamlit's explicit Dark theme, which tags an ancestor with data-theme="dark".
   Maroon is brightened (#cf4747) so it has enough contrast on the dark canvas. */
@media (prefers-color-scheme: dark) {
    :root {
        --uchi-maroon: #cf4747;
        --uchi-maroon-dark: #a83838;
        --bg: #0f1115;
        --bg-rgb: 15, 17, 21;
        --sidebar-bg: #161922;
        --border: #2a2e38;
        --border-soft: #1f2330;
        --text: #e8e6e3;
        --muted: #9a978f;
        --user-bubble: #2a2531;
        --surface: #1a1d24;
        --surface-hover: #22252e;
        --surface-alt: #1e222b;
        --surface-active: #2a2530;
        --expander-bg: #161922;
        --snippet-text: #cfccc6;
    }
}
html[data-theme="dark"],
[data-theme="dark"],
.stApp[data-theme="dark"] {
    --uchi-maroon: #cf4747;
    --uchi-maroon-dark: #a83838;
    --bg: #0f1115;
    --bg-rgb: 15, 17, 21;
    --sidebar-bg: #161922;
    --border: #2a2e38;
    --border-soft: #1f2330;
    --text: #e8e6e3;
    --muted: #9a978f;
    --user-bubble: #2a2531;
    --surface: #1a1d24;
    --surface-hover: #22252e;
    --surface-alt: #1e222b;
    --surface-active: #2a2530;
    --expander-bg: #161922;
    --snippet-text: #cfccc6;
}
</style>
"""


SUGGESTED_PROMPTS = [
    "What are the career outcomes of MS-ADS graduates?",
    "How much does the MS-ADS program cost?",
    "What are the application requirements?",
    "In-person vs. online — what's the difference?",
]


@st.cache_resource(show_spinner="Loading vector store and chains...")
def _load_chains():
    vectorstore = build_or_load_vectorstore()
    llm = _chat_llm()
    return {
        "llm": llm,
        "retrieval_chain": build_history_aware_retrieval_chain(vectorstore, llm),
    }


def _render_sources(docs) -> None:
    # Sources are already deduped and ordered by the chain, so the index here
    # matches the [N] citation numbers in the answer text.
    for i, d in enumerate(docs, start=1):
        url = d.metadata.get("url", "") or ""
        title = d.metadata.get("page_title", "") or "Untitled"
        section = d.metadata.get("section_title", "") or ""

        snippet = (d.page_content or "").strip()
        if len(snippet) > 320:
            snippet = snippet[:320].rsplit(" ", 1)[0] + "…"

        parts = ['<div class="source-card">']
        parts.append(f'<div class="title">[{i}] {html.escape(title)}</div>')
        if section:
            parts.append(f'<div class="section">{html.escape(section)}</div>')
        if snippet:
            parts.append(f'<div class="snippet">{html.escape(snippet)}</div>')
        if url:
            safe_url = html.escape(url, quote=True)
            parts.append(f'<a href="{safe_url}" target="_blank" rel="noopener">{html.escape(url)}</a>')
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)


def _render_message(msg: dict) -> None:
    avatar = ":material/school:" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                _render_sources(msg["sources"])


def _history_for_chain(messages: list[dict]) -> list[dict]:
    """Strip non-essential fields before sending to the condense chain."""
    return [{"role": m["role"], "content": m.get("content", "")} for m in messages]


def _answer(query: str, chains, history: list[dict]) -> dict:
    result = answer_with_history(
        query,
        chat_history=_history_for_chain(history),
        llm=chains["llm"],
        retrieval_chain=chains["retrieval_chain"],
    )
    return {"role": "assistant", "content": result["answer"], "sources": result["sources"]}


# ================== RENDER ==================
st.markdown(_CSS, unsafe_allow_html=True)
st.logo("assets/uchicago_logo.png", size="large")


def _new_chat() -> str:
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"id": cid, "title": "New chat", "messages": []}
    st.session_state.chat_order.insert(0, cid)
    st.session_state.current_chat_id = cid
    st.session_state.pending_query = None
    return cid


def _submit(query: str) -> None:
    chat = st.session_state.chats[st.session_state.current_chat_id]
    chat["messages"].append({"role": "user", "content": query})
    if chat["title"] == "New chat":
        chat["title"] = query if len(query) <= 36 else query[:33] + "…"
    st.session_state.pending_query = query


# ---- state bootstrap ----
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.chat_order = []
    _new_chat()
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if st.session_state.current_chat_id not in st.session_state.chats:
    # fallback in case state got out of sync
    if st.session_state.chat_order:
        st.session_state.current_chat_id = st.session_state.chat_order[0]
    else:
        _new_chat()

current = st.session_state.chats[st.session_state.current_chat_id]
messages = current["messages"]


# ---- sidebar ----
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
          <div class="product">MS-ADS Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("+ New chat", use_container_width=True, key="new_chat_btn", type="primary"):
        # Avoid stacking empty chats — if current is already empty, just stay on it.
        if current["messages"]:
            _new_chat()
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Chats</div>', unsafe_allow_html=True)

    non_empty = [cid for cid in st.session_state.chat_order if st.session_state.chats[cid]["messages"]]
    # Always include the active chat even if empty, so the user can see they're on it.
    visible = list(dict.fromkeys([st.session_state.current_chat_id, *non_empty]))

    if not any(st.session_state.chats[cid]["messages"] for cid in visible):
        st.markdown(
            '<div class="sidebar-empty-note">No chats yet</div>',
            unsafe_allow_html=True,
        )
    else:
        for cid in visible:
            chat = st.session_state.chats[cid]
            is_active = cid == st.session_state.current_chat_id
            label = chat["title"] or "New chat"
            if is_active:
                label = f"● {label}"
            btn_type = "tertiary" if is_active else "secondary"
            if st.button(label, key=f"chat_{cid}", use_container_width=True, type=btn_type):
                if not is_active:
                    st.session_state.current_chat_id = cid
                    st.session_state.pending_query = None
                    st.rerun()


chains = _load_chains()

# ---- main chat area ----
if not messages and not st.session_state.pending_query:
    st.markdown(
        """
        <div class="empty-state">
          <div class="emoji">🎓</div>
          <h1>MS-ADS Assistant</h1>
          <p>Ask anything about UChicago's Master's in Applied Data Science.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"suggest_{i}", use_container_width=True):
                _submit(prompt)
                st.rerun()
else:
    for msg in messages:
        _render_message(msg)

# Resolve a pending query (user message already appended; generate answer).
if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None
    target_chat = current  # lock to the chat that owned the submission
    # Prior turns = everything in this chat *except* the user message we're about to answer.
    prior_history = target_chat["messages"][:-1]
    with st.chat_message("assistant", avatar=":material/school:"):
        with st.spinner("Thinking…"):
            reply = _answer(q, chains, prior_history)
        st.markdown(reply["content"])
        if reply["sources"]:
            with st.expander(f"Sources ({len(reply['sources'])})"):
                _render_sources(reply["sources"])
    target_chat["messages"].append(reply)

# Sticky chat input.
submitted = st.chat_input("Message MS-ADS Assistant…")
if submitted:
    _submit(submitted)
    st.rerun()
