"""Paths and settings shared by the Day 7 pipeline."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QDRANT_PATH = REPO / 'day6_store' / 'qdrant_storage'
COLLECTION_STRATEGY = 'C'  # Day 6 selected strategy: by_paragraph <=1000 chars, 195 chunks
DATA_DIR = REPO / 'data' / 'day7'
CHUNKS_PATH = DATA_DIR / 'chunks.json'
QUESTIONS_PATH = DATA_DIR / 'questions.json'
RESULTS_DIR = REPO / 'results'
LLM_CACHE_DIR = RESULTS_DIR / 'llm_cache'


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(REPO / '.env')


def llm_settings() -> tuple[str, str]:
    """Return (provider, model) for the agent LLM; configurable in .env."""
    provider = os.getenv('LLM_PROVIDER', 'gemini').lower()
    default = 'gemini-3.1-flash-lite' if provider == 'gemini' else 'gpt-4o-mini'
    return provider, os.getenv('LLM_MODEL', default)


def open_day6(qdrant_path: Path = QDRANT_PATH):
    """Open the real Day 6 embedder and a Qdrant store (same model/dimension as the index)."""
    load_env()
    from src.day6.embeddings import MyEmbedder
    from src.day6.store import QdrantVectorStore
    if not qdrant_path.exists():
        raise FileNotFoundError(f'Missing Qdrant store at {qdrant_path}; see README')
    embedder = MyEmbedder()
    store = QdrantVectorStore(qdrant_path, dimensions=embedder.dimensions)
    return embedder, store
