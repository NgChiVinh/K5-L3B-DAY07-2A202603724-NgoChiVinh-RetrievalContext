"""Named retrieval/agent configurations and factories.

Retrieval methods (each benchmarked separately on the original question + user filter):
  dense   — Day 6 Qdrant strategy C (gemini-embedding-001, cosine)
  bm25    — BM25 syllable+bigram over the same 195 chunks
  hybrid  — RRF(dense, bm25), k=60, 20 candidates each
  dense_rerank / hybrid_rerank — LLM rerank of top-15 candidates
Agent configs live in AGENT_CONFIGS (see report for hypotheses).
"""
from __future__ import annotations

import json
from typing import Any

from . import config
from .retrievers import BM25Retriever, DenseRetriever, HybridRRF, LLMRerank

RETRIEVAL_METHODS = ['dense', 'bm25', 'hybrid', 'dense_rerank', 'hybrid_rerank']

AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    # Baseline: one dense tool, small context.
    'baseline_dense': {'tools': ['search_dense'], 'top_k': 3, 'budget_chars': 3000, 'max_searches': 3, 'neighbors': False},
    # Method comparison inside the agent.
    'agent_bm25': {'tools': ['search_keyword'], 'top_k': 3, 'budget_chars': 3000, 'max_searches': 3, 'neighbors': False},
    'agent_hybrid': {'tools': ['search_hybrid'], 'top_k': 3, 'budget_chars': 3000, 'max_searches': 3, 'neighbors': False},
    # Improvement 1: hybrid + LLM rerank, top_k 5.
    'improve_rerank': {'tools': ['search_hybrid_rerank'], 'top_k': 5, 'budget_chars': 5000, 'max_searches': 3, 'neighbors': False},
    # Improvement 2: agent chooses among dense/keyword tools, more searches, larger budget + neighbours.
    'improve_multitool': {'tools': ['search_dense', 'search_keyword'], 'top_k': 5, 'budget_chars': 6000,
                          'max_searches': 4, 'neighbors': True},
}


class Context:
    """Lazily opens the real Day 6 store/embedder and the LLM once per process."""

    def __init__(self):
        config.load_env()
        self.chunks = json.loads(config.CHUNKS_PATH.read_text(encoding='utf-8'))
        self._dense = self._bm25 = self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLMClient
            provider, model = config.llm_settings()
            self._llm = LLMClient(provider, model, cache_dir=config.LLM_CACHE_DIR)
        return self._llm

    @property
    def dense(self):
        if self._dense is None:
            embedder, store = config.open_day6()
            self._dense = DenseRetriever(store, embedder, strategy=config.COLLECTION_STRATEGY)
        return self._dense

    @property
    def bm25(self):
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.chunks)
        return self._bm25

    def retriever(self, name: str):
        if name == 'dense':
            return self.dense
        if name == 'bm25':
            return self.bm25
        if name == 'hybrid':
            return HybridRRF([self.dense, self.bm25], k=60, candidates=20)
        if name == 'dense_rerank':
            return LLMRerank(self.dense, self.llm, candidates=15)
        if name == 'hybrid_rerank':
            return LLMRerank(self.retriever('hybrid'), self.llm, candidates=15)
        raise KeyError(name)


TOOL_TO_RETRIEVER = {'search_dense': 'dense', 'search_keyword': 'bm25', 'search_hybrid': 'hybrid',
                     'search_hybrid_rerank': 'hybrid_rerank'}
