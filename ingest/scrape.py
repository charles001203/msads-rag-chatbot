"""BFS crawler that caches raw HTML pages to disk.

Crawls the UChicago MSADS website within the URL allowlist defined in config.py,
skipping denylisted paths and non-HTML extensions. Output: one HTML file per
fetched URL under data/raw/, plus a manifest.json describing the crawl.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import config


def _init_logger() -> logging.Logger:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scrape")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(config.SCRAPE_LOG_PATH, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(sh)
    return logger


def _url_cache_path(url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return config.RAW_DIR / f"{digest}.html"


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    if url.endswith("/"):
        return url
    # keep trailing slash on paths without a file extension
    parsed = urlparse(url)
    path = parsed.path
    if "." not in path.rsplit("/", 1)[-1]:
        return url + "/"
    return url


def is_allowed(url: str) -> bool:
    if not url.startswith("http"):
        return False
    if any(sub in url for sub in config.URL_DENYLIST_SUBSTRINGS):
        return False
    lower = url.lower()
    if any(lower.endswith(ext) for ext in config.DENIED_EXTENSIONS):
        return False
    return any(url.startswith(pre) for pre in config.URL_ALLOWLIST_PREFIXES)


def fetch(url: str, logger: logging.Logger) -> tuple[int, str | None]:
    headers = {"User-Agent": config.USER_AGENT, "Accept": "text/html"}
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url, headers=headers, timeout=config.REQUEST_TIMEOUT_SECONDS
            )
            if resp.status_code >= 500:
                raise requests.RequestException(f"server error {resp.status_code}")
            content_type = resp.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                logger.info("skip non-html %s (%s)", url, content_type)
                return resp.status_code, None
            return resp.status_code, resp.text
        except requests.RequestException as exc:
            wait = config.RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "fetch failed (attempt %d/%d) %s: %s — retrying in %.1fs",
                attempt, config.MAX_RETRIES, url, exc, wait,
            )
            time.sleep(wait)
    logger.error("giving up on %s", url)
    return 0, None


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: set[str] = set()
    for a in soup.find_all("a", href=True):
        raw = a["href"].strip()
        if not raw or raw.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, raw)
        absolute = normalize_url(absolute)
        if is_allowed(absolute):
            links.add(absolute)
    return sorted(links)


def crawl() -> list[dict]:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger = _init_logger()
    queue: deque[tuple[str, int]] = deque(
        (normalize_url(u), 0) for u in config.SEED_URLS
    )
    visited: set[str] = set()
    manifest: list[dict] = []

    pbar = tqdm(desc="crawl", unit="page")
    while queue:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        if not is_allowed(url):
            continue

        cache_path = _url_cache_path(url)
        html: str | None = None
        status = 0
        from_cache = cache_path.exists()

        if from_cache:
            html = cache_path.read_text(encoding="utf-8", errors="replace")
            status = 200
            logger.info("cache hit %s", url)
        else:
            status, html = fetch(url, logger)
            if html is not None:
                cache_path.write_text(html, encoding="utf-8")
            time.sleep(config.REQUEST_DELAY_SECONDS)

        manifest.append({
            "url": url,
            "depth": depth,
            "status": status,
            "cache_file": cache_path.name if html is not None else None,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "bytes": len(html) if html else 0,
            "from_cache": from_cache,
        })
        pbar.update(1)
        pbar.set_postfix_str(url[-60:])

        if html is None:
            continue
        if depth >= config.MAX_DEPTH:
            continue
        for link in extract_links(html, url):
            if link not in visited:
                queue.append((link, depth + 1))

    pbar.close()
    config.MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("crawl complete: %d pages", len(manifest))
    return manifest


if __name__ == "__main__":
    crawl()
