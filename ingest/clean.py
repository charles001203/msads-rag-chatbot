"""Normalize, dedupe, and enrich extracted records into the final knowledge base.

Input: data/interim/records.json (from extract.py)
Output: data/processed/knowledge_base.json + knowledge_base_stats.json
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable

import config


_WS_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\u00a0", " ")
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def _jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _canonical_url(urls: Iterable[str]) -> str:
    # shortest URL wins (closest to the hub)
    return min(urls, key=len)


def _normalize_money_and_pct(text: str) -> str:
    # collapse "$65000" -> "$65,000" when it looks like dollars
    def _comma(match: re.Match) -> str:
        num = match.group(1)
        return "$" + f"{int(num):,}"
    text = re.sub(r"\$(\d{4,})(?!\d)", _comma, text)
    # strip space between number and %
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    return text


def load_records() -> list[dict]:
    if not config.INTERIM_RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"interim records not found at {config.INTERIM_RECORDS_PATH}; "
            "run extract.py first"
        )
    return json.loads(config.INTERIM_RECORDS_PATH.read_text(encoding="utf-8"))


def clean(records: list[dict]) -> tuple[list[dict], dict]:
    # 1. normalize text
    for r in records:
        r["content"] = _normalize_money_and_pct(normalize_text(r["content"]))
        r["section_title"] = normalize_text(r.get("section_title", ""))
        r["page_title"] = normalize_text(r.get("page_title", ""))

    # drop empty / too-short after normalization
    records = [r for r in records if len(r["content"]) >= config.MIN_RECORD_CHARS]

    # 2. exact-dup boilerplate removal based on content hash appearing on N+ pages
    content_to_urls: dict[str, set[str]] = defaultdict(set)
    for r in records:
        content_to_urls[r["content"]].add(r["url"])

    boilerplate_contents = {
        c for c, urls in content_to_urls.items()
        if len(urls) >= config.BOILERPLATE_PAGE_THRESHOLD
    }
    dropped_boilerplate: list[dict] = []
    kept: list[dict] = []
    seen_canonical: set[str] = set()
    for r in records:
        if r["content"] in boilerplate_contents:
            canonical = _canonical_url(content_to_urls[r["content"]])
            key = (r["content"], canonical)
            if r["url"] == canonical and key not in seen_canonical:
                seen_canonical.add(key)
                r.setdefault("also_on", sorted(content_to_urls[r["content"]] - {canonical}))
                kept.append(r)
            else:
                dropped_boilerplate.append({
                    "url": r["url"],
                    "section_title": r["section_title"],
                    "reason": "boilerplate",
                    "appears_on": sorted(content_to_urls[r["content"]]),
                })
        else:
            kept.append(r)
    records = kept

    # 3. near-duplicate consolidation (>90% jaccard)
    # compare only within same section_title to keep it cheap
    by_title: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_title[r["section_title"].lower()].append(i)
    to_drop: set[int] = set()
    near_dup_report: list[dict] = []
    for title, idxs in by_title.items():
        if len(idxs) < 2:
            continue
        for a_pos in range(len(idxs)):
            ia = idxs[a_pos]
            if ia in to_drop:
                continue
            for b_pos in range(a_pos + 1, len(idxs)):
                ib = idxs[b_pos]
                if ib in to_drop:
                    continue
                ra, rb = records[ia], records[ib]
                if _jaccard(ra["content"], rb["content"]) >= config.NEAR_DUPLICATE_JACCARD:
                    # keep the one with shorter canonical URL, merge cross-refs
                    keep_idx, drop_idx = (ia, ib) if len(ra["url"]) <= len(rb["url"]) else (ib, ia)
                    keeper = records[keep_idx]
                    loser = records[drop_idx]
                    refs = set(keeper.get("also_on", []))
                    refs.add(loser["url"])
                    keeper["also_on"] = sorted(refs)
                    to_drop.add(drop_idx)
                    near_dup_report.append({
                        "kept_url": keeper["url"],
                        "dropped_url": loser["url"],
                        "section_title": keeper["section_title"],
                    })
    records = [r for i, r in enumerate(records) if i not in to_drop]

    # 4. breadcrumb-prefixed content + FAQ formatting + record_id
    for i, r in enumerate(records):
        body = r["content"]
        if r.get("content_type") == "faq":
            body = f"Q: {r['section_title']}\nA: {body}"
        prefix = f"[Page: {r['page_title']}] [Section: {r['section_title']}]\n\n"
        r["content"] = prefix + body
        r["record_id"] = f"rec_{i:05d}"
        r["char_count"] = len(r["content"])
        r["word_count"] = len(r["content"].split())

    # 5. stats
    by_type = Counter(r["content_type"] for r in records)
    by_page = Counter(r["url"] for r in records)
    total_words = sum(r["word_count"] for r in records)
    stats = {
        "total_records": len(records),
        "total_words": total_words,
        "by_content_type": dict(by_type),
        "records_per_page": dict(by_page.most_common()),
        "dropped_boilerplate_count": len(dropped_boilerplate),
        "dropped_boilerplate_sample": dropped_boilerplate[:10],
        "near_duplicates_merged": near_dup_report[:20],
    }
    return records, stats


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    cleaned, stats = clean(records)
    config.KNOWLEDGE_BASE_PATH.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config.STATS_PATH.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(cleaned)} records -> {config.KNOWLEDGE_BASE_PATH}")
    print(f"stats -> {config.STATS_PATH}")


if __name__ == "__main__":
    main()
