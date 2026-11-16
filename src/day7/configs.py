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
from .retrievers import BM25Retriever, DenseRetriever, HybridRRF, LabelAwareRetriever, LLMRerank

RETRIEVAL_METHODS = ['dense', 'bm25', 'hybrid', 'dense_rerank', 'hybrid_rerank', 'hybrid_label']

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
    # Advanced corpus (dev failures d02/d10/d11/d12/du1/du2):
    # A: queries naming 'Bảng/Hình X.Y' get that visual chunk first (label inferred from the question itself).
    'improve_label': {'tools': ['search_hybrid_label'], 'top_k': 5, 'budget_chars': 6000, 'max_searches': 4, 'neighbors': False},
    # A + B: same retrieval + stricter answer rules (abstain marker first, no look-alike metrics).
    'improve_label_strict': {'tools': ['search_hybrid_label'], 'top_k': 5, 'budget_chars': 6000, 'max_searches': 4,
                             'neighbors': False, 'strict_prompt': True},
    # C: A + B with a stronger agent model (same provider) for multi-part checking and abstention.
    'improve_label_strict_41': {'tools': ['search_hybrid_label'], 'top_k': 5, 'budget_chars': 6000, 'max_searches': 4,
                                'neighbors': False, 'strict_prompt': True, 'llm_model': 'gpt-4.1'},
    # E: C + larger budget for long tables and a per-search cap so later searches are never starved
    #    (dev d10: Bảng 2.3 ranked #1 in search 2 but skipped because search 1 filled 5160/6000 chars).
    'improve_label_strict_41_budget': {'tools': ['search_hybrid_label'], 'top_k': 5, 'budget_chars': 10000,
                                       'per_search_chars': 4000, 'max_searches': 4, 'neighbors': False,
                                       'strict_prompt': True, 'llm_model': 'gpt-4.1'},
}


CORPORA = {
    # name: (chunks, questions, qdrant path, collection strategy, results subdir, doc title for the prompt)
    'day6': (config.CHUNKS_PATH, config.QUESTIONS_PATH, config.QDRANT_PATH, config.COLLECTION_STRATEGY, '',
             'Điểm lại tháng 8/2024: Vươn tới tầm cao mới trên thị trường vốn'),
    'adv': (config.REPO / 'data' / 'adv' / 'chunks.json', config.REPO / 'data' / 'adv' / 'questions.json',
            config.REPO / 'adv_store' / 'qdrant', 'ADV', 'adv',
            'Việt Nam 2045 – Tăng trưởng Xanh hơn: Con đường hướng tới tương lai bền vững'),
}


class Context:
    """Lazily opens the real store/embedder (same gemini-embedding-001 3072d) and the LLM once per process."""

    def __init__(self, corpus: str = 'day6', model_override: str | None = None):
        self.model_override = model_override
        config.load_env()
        self.corpus = corpus
        (self.chunks_path, self.questions_path, self.qdrant_path, self.strategy,
         self.results_subdir, self.doc_title) = CORPORA[corpus]
        self.chunks = json.loads(self.chunks_path.read_text(encoding='utf-8'))
        self._dense = self._bm25 = self._llm = None

    @property
    def results_dir(self):
        return config.RESULTS_DIR / self.results_subdir if self.results_subdir else config.RESULTS_DIR

    def questions(self, split: str) -> list[dict[str, Any]]:
        return [q for q in json.loads(self.questions_path.read_text(encoding='utf-8')) if q['split'] == split]

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLMClient
            provider, model = config.llm_settings('ADV_' if self.corpus == 'adv' else '')
            self._llm = LLMClient(provider, self.model_override or model, cache_dir=config.LLM_CACHE_DIR)
        return self._llm

    @property
    def dense(self):
        if self._dense is None:
            embedder, store = config.open_day6(self.qdrant_path)
            self._dense = DenseRetriever(store, embedder, strategy=self.strategy)
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
        if name == 'hybrid_label':
            return LabelAwareRetriever(self.retriever('hybrid'), self.chunks)
        if name == 'hybrid_rerank':
            return LLMRerank(self.retriever('hybrid'), self.llm, candidates=15)
        raise KeyError(name)


TOOL_TO_RETRIEVER = {'search_dense': 'dense', 'search_keyword': 'bm25', 'search_hybrid': 'hybrid',
                     'search_hybrid_rerank': 'hybrid_rerank', 'search_hybrid_label': 'hybrid_label'}
