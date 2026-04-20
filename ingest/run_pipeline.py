"""End-to-end orchestrator: scrape -> extract -> clean."""
from __future__ import annotations

from ingest import clean, extract, scrape


def main() -> None:
    print("=== stage 1/3: scrape ===")
    manifest = scrape.crawl()
    print(f"crawled {len(manifest)} pages")

    print("\n=== stage 2/3: extract ===")
    records = extract.extract_all()
    print(f"extracted {len(records)} records")

    print("\n=== stage 3/3: clean ===")
    clean.main()


if __name__ == "__main__":
    main()
