# Copied unchanged from Day 6 repo (NgChiVinh/K5-L3B-DAY06-...-DataIngestion, starter/src/embeddings.py) to reuse the same embedder and Qdrant store.
"""Real embedding provider adapter (Gemini API / OpenAI API / Local SentenceTransformers)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int
    provider: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_queries(self, questions: list[str]) -> list[list[float]]:
        if not questions:
            return []
        if self.provider == "gemini":
            vecs = self._embed_gemini_with_retry(questions, task_type="RETRIEVAL_QUERY")
            self.total_embedded_texts += len(questions)
            self.total_embedded_chars += sum(len(q) for q in questions)
            return vecs
        return [self.embed_query(q) for q in questions]

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Real embedding provider adapter; never falls back to mock or random vectors.

    Includes deterministic on-disk vector caching and Free-Tier 429 rate-limit backoff
    (100 embed items/minute on Gemini Free Tier) plus error event logging.
    """

    def __init__(
        self,
        model_name: str | None = None,
        provider: str | None = None,
        batch_size: int = 80,
        cache_path: Path | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.total_embedded_texts = 0
        self.total_embedded_chars = 0
        self.api_calls = 0
        self.rate_limit_events: list[dict[str, Any]] = []

        repo_root = Path(__file__).resolve().parents[2]
        self.cache_path = cache_path or (repo_root / "results" / "embedding_cache.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        configured_model = model_name or os.getenv("EMBEDDING_MODEL") or ""

        if provider is None:
            if gemini_key:
                provider = "gemini"
            elif openai_key:
                provider = "openai"
            elif openrouter_key:
                provider = "openrouter"
            elif configured_model.startswith("sentence-transformers/") or configured_model.startswith("local:"):
                provider = "local"
            else:
                raise RuntimeError(
                    "No embedding API key or local model configured. "
                    "Please set GEMINI_API_KEY (or OPENAI_API_KEY / local sentence-transformers model) in starter/.env. "
                    "Silent fallback to mock/random vectors is strictly disabled per RULES.md."
                )

        self.provider = provider.lower()

        if self.provider == "gemini":
            if not gemini_key:
                raise RuntimeError("GEMINI_API_KEY is missing in environment or starter/.env.")
            from google import genai

            self._client = genai.Client(api_key=gemini_key)
            self.model_name = configured_model or "gemini-embedding-001"
            probe_vec = self._embed_gemini_with_retry(["Kiểm tra kết nối model embedding"], task_type="RETRIEVAL_QUERY")[0]
            self.dimensions = len(probe_vec)

        elif self.provider in ("openai", "openrouter"):
            from openai import OpenAI

            api_key = openai_key if self.provider == "openai" else openrouter_key
            if not api_key:
                raise RuntimeError(f"API key for {self.provider} is missing.")
            base_url = "https://openrouter.ai/api/v1" if self.provider == "openrouter" else None
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self.model_name = configured_model or "text-embedding-3-small"
            probe = self._client.embeddings.create(model=self.model_name, input=["Kiểm tra kết nối"])
            self.dimensions = len(probe.data[0].embedding)
            self.api_calls += 1

        elif self.provider == "local":
            from sentence_transformers import SentenceTransformer

            clean_name = configured_model.replace("local:", "") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.model_name = clean_name
            self._st_model = SentenceTransformer(self.model_name)
            probe_vec = self._st_model.encode(["Kiểm tra kết nối"], normalize_embeddings=True)
            self.dimensions = int(probe_vec.shape[1])

        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _cache_key(self, task_type: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.provider}:{self.model_name}:{task_type}:{digest}"

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")

    def _embed_gemini_with_retry(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        for idx, t in enumerate(texts):
            ck = self._cache_key(task_type, t)
            if ck in self._cache:
                results[idx] = self._cache[ck]
            else:
                missing_indices.append(idx)

        if not missing_indices:
            return [r for r in results if r is not None]

        for start in range(0, len(missing_indices), self.batch_size):
            batch_idxs = missing_indices[start : start + self.batch_size]
            batch_texts = [texts[i] for i in batch_idxs]

            for attempt in range(10):
                try:
                    resp = self._client.models.embed_content(
                        model=self.model_name,
                        contents=batch_texts,
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    self.api_calls += 1
                    break
                except Exception as exc:
                    err_str = str(exc)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        m = re.search(r"retry in ([0-9.]+)s", err_str, re.IGNORECASE)
                        wait_sec = float(m.group(1)) + 4.0 if m else 62.0
                        event = {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "provider": "gemini",
                            "model": self.model_name,
                            "batch_size": len(batch_texts),
                            "attempt": attempt + 1,
                            "wait_seconds": round(wait_sec, 2),
                            "reason": "429 RESOURCE_EXHAUSTED (Free Tier 100 embed items/min limit)",
                        }
                        self.rate_limit_events.append(event)
                        print(
                            f"  [Rate Limit 429] Gemini Free Tier 100 items/min quota reached; "
                            f"sleeping {wait_sec:.1f}s before retry (attempt {attempt + 1}/10)...",
                            flush=True,
                        )
                        time.sleep(wait_sec)
                    else:
                        if attempt >= 3:
                            raise RuntimeError(f"Gemini embed_content failed: {exc}") from exc
                        time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError("Exceeded max retries for Gemini embed_content.")

            if not resp.embeddings or len(resp.embeddings) != len(batch_texts):
                raise RuntimeError("Gemini returned mismatched number of embeddings.")

            for i_pos, emb in zip(batch_idxs, resp.embeddings):
                vec = list(emb.values)
                results[i_pos] = vec
                ck = self._cache_key(task_type, texts[i_pos])
                self._cache[ck] = vec

            self._save_cache()

        return [r for r in results if r is not None]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self.provider == "gemini":
            all_vectors = self._embed_gemini_with_retry(texts, task_type="RETRIEVAL_DOCUMENT")

        elif self.provider in ("openai", "openrouter"):
            all_vectors = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                resp = self._client.embeddings.create(model=self.model_name, input=batch)
                for item in resp.data:
                    all_vectors.append(list(item.embedding))
                self.api_calls += 1

        elif self.provider == "local":
            vecs = self._st_model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
            all_vectors = [row.tolist() for row in vecs]

        self.total_embedded_texts += len(texts)
        self.total_embedded_chars += sum(len(t) for t in texts)
        return all_vectors

    def embed_queries(self, questions: list[str]) -> list[list[float]]:
        if not questions:
            return []
        if self.provider == "gemini":
            vecs = self._embed_gemini_with_retry(questions, task_type="RETRIEVAL_QUERY")
            self.total_embedded_texts += len(questions)
            self.total_embedded_chars += sum(len(q) for q in questions)
            return vecs
        return [self.embed_query(q) for q in questions]

    def embed_query(self, question: str) -> list[float]:
        if not question or not question.strip():
            raise ValueError("Query question cannot be empty.")

        if self.provider == "gemini":
            vec = self._embed_gemini_with_retry([question], task_type="RETRIEVAL_QUERY")[0]

        elif self.provider in ("openai", "openrouter"):
            resp = self._client.embeddings.create(model=self.model_name, input=[question])
            vec = list(resp.data[0].embedding)
            self.api_calls += 1

        elif self.provider == "local":
            arr = self._st_model.encode([question], normalize_embeddings=True)
            vec = arr[0].tolist()

        self.total_embedded_texts += 1
        self.total_embedded_chars += len(question)
        return vec
