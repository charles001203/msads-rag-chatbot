"""Crawl policy and paths for the UChicago MSADS RAG knowledge base pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

MANIFEST_PATH = RAW_DIR / "manifest.json"
INTERIM_RECORDS_PATH = INTERIM_DIR / "records.json"
KNOWLEDGE_BASE_PATH = PROCESSED_DIR / "knowledge_base.json"
STATS_PATH = PROCESSED_DIR / "knowledge_base_stats.json"
SCRAPE_LOG_PATH = LOGS_DIR / "scrape.log"

BASE_HOST = "https://datascience.uchicago.edu"
MSADS_ROOT = f"{BASE_HOST}/education/masters-programs/ms-in-applied-data-science/"

SEED_URLS = [
    MSADS_ROOT,
    f"{MSADS_ROOT}in-person-program/",
    f"{MSADS_ROOT}online-program/",
    f"{MSADS_ROOT}mba-ms-joint-degree/",
    f"{MSADS_ROOT}course-progressions/",
    f"{MSADS_ROOT}faqs/",
    f"{MSADS_ROOT}how-to-apply/",
    f"{MSADS_ROOT}tuition-fees-aid/",
    f"{MSADS_ROOT}career-outcomes/",
    f"{MSADS_ROOT}capstone/",
    f"{MSADS_ROOT}instructors/",
    f"{MSADS_ROOT}student-profiles/",
    f"{MSADS_ROOT}events-deadlines/",
]

URL_ALLOWLIST_PREFIXES = (
    MSADS_ROOT,
    f"{BASE_HOST}/people/",
)

URL_DENYLIST_SUBSTRINGS = (
    "apply-psd.uchicago.edu",
    "/news/",
    "/events/",
    "/news-events/",
    "/category/",
    "/tag/",
    "/author/",
    "/wp-content/",
    "/wp-admin/",
    "/wp-login",
    "/feed",
    "?share=",
)

DENIED_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".mp4", ".mov", ".zip", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".ics",
)

REQUEST_TIMEOUT_SECONDS = 20
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
MAX_DEPTH = 3
USER_AGENT = (
    "UChicago-MSADS-RAG-Midterm-Bot/0.1 "
    "(coursework; charles.yu.1203@gmail.com)"
)

MIN_RECORD_CHARS = 30
BOILERPLATE_PAGE_THRESHOLD = 3
NEAR_DUPLICATE_JACCARD = 0.90

SOURCE_NAME = "uchicago-msads-website"

# RAG app settings
CHROMA_DIR = str(DATA_DIR / "chroma")
COLLECTION_NAME = "UChi_Midterm"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 4
NUM_MULTI_QUERIES = 4
NUM_CONDENSE_HISTORY_TURNS = 6
TOP_N_SOURCES = 5
