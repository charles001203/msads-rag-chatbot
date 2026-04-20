"""Parse cached HTML into structured, section-level records.

Each page becomes one or more records (one per heading-delimited section).
Lists and tables are preserved as structured text. FAQ accordions are split
into one record per Q&A pair with content_type="faq".
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm

import config

NON_CONTENT_SELECTORS = [
    "script", "style", "noscript", "nav", "header", "footer", "form",
    "[role=navigation]", "[role=banner]", "[role=contentinfo]",
    ".site-header", ".site-footer", ".breadcrumb", ".breadcrumbs",
    ".screen-reader-text", ".skip-link", ".menu", ".sub-menu",
    ".wp-block-navigation", ".cookie", ".cookies", ".announcement-bar",
    ".social-share", ".social-links",
]

MAIN_ROOT_SELECTORS = [
    "main", "article", "div.entry-content", "div.site-main",
    "div.main-content", "div#content", "div#main",
]

SECTION_HEADING_TAGS = ("h1", "h2", "h3")
MAX_SECTION_CHARS = 4000  # split oversize sections on paragraph boundaries


def load_manifest() -> list[dict]:
    if not config.MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"manifest not found at {config.MANIFEST_PATH}; run scrape.py first"
        )
    return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))


def _strip_non_content(soup: BeautifulSoup) -> None:
    for selector in NON_CONTENT_SELECTORS:
        for el in soup.select(selector):
            el.decompose()


def _find_main_root(soup: BeautifulSoup) -> Tag:
    for selector in MAIN_ROOT_SELECTORS:
        el = soup.select_one(selector)
        if el and el.get_text(strip=True):
            return el
    return soup.body or soup


def _text(el: Tag) -> str:
    return " ".join(el.get_text(" ", strip=True).split())


def _list_text(el: Tag) -> str:
    items = [f"- {_text(li)}" for li in el.find_all("li", recursive=False) if _text(li)]
    return "\n".join(items)


def _table_text(el: Tag) -> str:
    rows_out: list[str] = []
    headers: list[str] = []
    header_row = el.find("tr")
    if header_row:
        header_cells = header_row.find_all(["th", "td"])
        if all(c.name == "th" for c in header_cells) and header_cells:
            headers = [_text(c) for c in header_cells]
    for tr in el.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        if headers and all(c.name == "th" for c in cells):
            continue
        values = [_text(c) for c in cells]
        if headers and len(values) == len(headers):
            rows_out.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, values) if v))
        else:
            rows_out.append(" | ".join(v for v in values if v))
    return "\n".join(r for r in rows_out if r)


def _is_heading(el: Tag) -> bool:
    return isinstance(el, Tag) and el.name in SECTION_HEADING_TAGS


def _walk_blocks(root: Tag):
    """Yield top-level block elements in document order."""
    for child in root.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                yield ("text", text, None)
            continue
        if not isinstance(child, Tag):
            continue
        yield from _yield_from_tag(child)


def _yield_from_tag(tag: Tag):
    name = tag.name
    if name in SECTION_HEADING_TAGS:
        yield ("heading", _text(tag), name)
        return
    if name in ("p", "blockquote"):
        t = _text(tag)
        if t:
            yield ("prose", t, None)
        return
    if name in ("ul", "ol"):
        t = _list_text(tag)
        if t:
            yield ("list", t, None)
        return
    if name == "table":
        t = _table_text(tag)
        if t:
            yield ("table", t, None)
        return
    if name in ("details",):
        summary = tag.find("summary")
        q = _text(summary) if summary else ""
        # remove summary for body extraction
        if summary:
            summary.extract()
        body = _text(tag)
        if q or body:
            yield ("faq", (q, body), None)
        return
    # recurse into generic containers (div, section, article, aside)
    for child in tag.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text and len(text) > 1:
                yield ("prose", text, None)
        elif isinstance(child, Tag):
            yield from _yield_from_tag(child)


def _extract_accordion_items(main: Tag) -> list[tuple[str, str]]:
    """Pull Q&A-style pairs from the site's accordion widgets.

    The UChicago DSI site uses a consistent pattern: `.accordion__item`
    containers with `.accordion-title` for the question/heading and
    `.accordion__content` for the body. Also handles `<details>/<summary>`
    and a couple of WP-block variants as fallbacks.

    This destructively extracts accordions from `main` so downstream
    section-splitting doesn't see them again.
    """
    pairs: list[tuple[str, str]] = []

    # pattern A: site-native accordion__item
    for item in main.select(".accordion__item"):
        title_el = item.select_one(".accordion-title, .accordion__title")
        content_el = item.select_one(".accordion__content, .accordion-content")
        q = _text(title_el) if title_el else ""
        a = _text(content_el) if content_el else ""
        if q and a and len(a) >= 20:
            pairs.append((q, a))
        item.decompose()

    # pattern B: <details><summary>Q</summary>A</details>
    for det in main.find_all("details"):
        summary = det.find("summary")
        if summary:
            q = _text(summary)
            summary.extract()
            a = _text(det)
            if q and a:
                pairs.append((q, a))
        det.decompose()

    # pattern C: WP FAQ blocks
    for selector in [".wp-block-uagb-faq-child", ".faq-item"]:
        for item in main.select(selector):
            q_el = item.find(["h2", "h3", "h4", "h5", "button", "summary"])
            if not q_el:
                continue
            q = _text(q_el)
            q_el.extract()
            a = _text(item)
            if q and a and len(a) >= 20:
                pairs.append((q, a))
            item.decompose()

    return pairs


def _extract_sections(main: Tag, page_title: str) -> list[dict]:
    """Split the main content root into heading-delimited sections."""
    sections: list[dict] = []
    current_title = page_title
    current_parts: list[tuple[str, str]] = []  # (content_type, text)

    def flush():
        if not current_parts:
            return
        # aggregate
        text_parts = [t for _, t in current_parts]
        content = "\n\n".join(text_parts).strip()
        if not content:
            return
        # content_type is the dominant non-prose type, else prose
        types = {ct for ct, _ in current_parts}
        if "table" in types:
            ctype = "table"
        elif "list" in types and "prose" not in types:
            ctype = "list"
        else:
            ctype = "prose"
        sections.append({
            "section_title": current_title,
            "content": content,
            "content_type": ctype,
        })

    for kind, payload, _meta in _walk_blocks(main):
        if kind == "heading":
            flush()
            current_title = payload or page_title
            current_parts = []
        elif kind == "faq":
            q, a = payload
            flush()
            sections.append({
                "section_title": q or current_title,
                "content": a,
                "content_type": "faq",
                "_faq_q": q,
            })
            current_parts = []
            current_title = page_title
        else:
            current_parts.append((kind, payload))
    flush()
    return sections


def _split_oversize(content: str, max_chars: int = MAX_SECTION_CHARS) -> list[str]:
    """Split a section on paragraph boundaries if it exceeds max_chars.

    Used as a safety net for pages that have weak heading structure — we'd
    rather hand the embedder 5 focused chunks than one 30kB blob.
    """
    if len(content) <= max_chars:
        return [content]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in content.split("\n\n"):
        p_len = len(para)
        if current and current_len + p_len + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = p_len
        else:
            current.append(para)
            current_len += p_len + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def extract_one(cache_path: Path, url: str) -> list[dict]:
    if not cache_path.exists():
        return []
    html = cache_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    _strip_non_content(soup)

    title_tag = soup.find("title")
    page_title = _text(title_tag) if title_tag else url
    for sep in [" | ", " – ", " - "]:
        if sep in page_title:
            page_title = page_title.split(sep)[0].strip()
            break

    main = _find_main_root(soup)
    scraped_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []

    # 1. Pull out accordion-style Q&A first (destructive) so sections don't
    #    re-catch them. This fires on any page that uses the pattern — FAQs,
    #    Course Progressions (course listings), program comparison pages, etc.
    accordion_pairs = _extract_accordion_items(main)
    for q, a in accordion_pairs:
        records.append({
            "url": url,
            "page_title": page_title,
            "section_title": q,
            "content": a,
            "content_type": "faq",
            "scraped_at": scraped_at,
            "source": config.SOURCE_NAME,
        })

    # 2. Heading-delimited sections from what remains
    sections = _extract_sections(main, page_title)
    for idx, sec in enumerate(sections):
        chunks = _split_oversize(sec["content"])
        for chunk_idx, chunk in enumerate(chunks):
            records.append({
                "url": url,
                "page_title": page_title,
                "section_title": sec["section_title"] if len(chunks) == 1
                    else f"{sec['section_title']} (part {chunk_idx + 1}/{len(chunks)})",
                "section_index": idx,
                "content": chunk,
                "content_type": sec["content_type"],
                "scraped_at": scraped_at,
                "source": config.SOURCE_NAME,
            })
    return records


def extract_all() -> list[dict]:
    manifest = load_manifest()
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    for entry in tqdm(manifest, desc="extract", unit="page"):
        if not entry.get("cache_file") or entry.get("status") != 200:
            continue
        cache_path = config.RAW_DIR / entry["cache_file"]
        recs = extract_one(cache_path, entry["url"])
        recs = [r for r in recs if len(r["content"]) >= config.MIN_RECORD_CHARS]
        all_records.extend(recs)
    config.INTERIM_RECORDS_PATH.write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return all_records


if __name__ == "__main__":
    records = extract_all()
    print(f"extracted {len(records)} records")
