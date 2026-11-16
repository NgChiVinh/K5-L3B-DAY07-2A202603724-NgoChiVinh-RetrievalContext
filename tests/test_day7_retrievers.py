"""Offline checks for lexical/hybrid/rerank retrievers; fixed fakes stand in for Qdrant and the LLM."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.day7.retrievers import BM25Retriever, DenseRetriever, HybridRRF, LLMRerank, tokenize


def chunk(cid, text, page=1, section='a'):
    return {'id': cid, 'text': text, 'source_id': 'doc', 'source': 'a.pdf', 'page': page,
            'metadata': {'section': section}}


CHUNKS = [chunk('c1', 'Lạm phát tăng do giá lương thực.'),
          chunk('c2', 'Tỷ lệ nợ xấu của hệ thống ngân hàng tăng lên 4,5%.', section='b'),
          chunk('c3', 'Ngân hàng Nhà nước giữ nguyên lãi suất.')]


class FixedRetriever:
    def __init__(self, ids): self.ids = ids
    def search(self, query, *, filters=None, top_k=3):
        return [dict(chunk(i, i), score=1.0, rank=r, method='fixed') for r, i in enumerate(self.ids[:top_k], 1)]


class FakeStore:
    def search(self, strategy, vec, top_k=3, filters=None):
        self.args = (strategy, top_k, filters)
        return [{'rank': 1, 'chunk_id': 'c9', 'score': 0.8, 'text': 'x',
                 'metadata': {'document_id': 'doc', 'source': 'a.pdf', 'page': 7, 'section': 'a', 'chunk_id': 'c9'}}]


class FakeEmbedder:
    def embed_query(self, q): return [0.1, 0.2]


class FakeLLM:
    def __init__(self, text): self.text = text; self.prompts = []
    def complete(self, system, user):
        self.prompts.append(user); return {'text': self.text, 'usage': {'input_tokens': 1, 'output_tokens': 1}}


class RetrieverTests(unittest.TestCase):
    def test_tokenize_adds_syllable_bigrams(self):
        self.assertEqual(tokenize('Nợ xấu'), ['nợ', 'xấu', 'nợ_xấu'])

    def test_bm25_ranks_exact_phrase_first(self):
        hits = BM25Retriever(CHUNKS).search('tỷ lệ nợ xấu', filters=None, top_k=2)
        self.assertEqual(hits[0]['id'], 'c2')
        self.assertEqual(hits[0]['rank'], 1)
        self.assertEqual(hits[0]['method'], 'bm25')

    def test_bm25_honors_metadata_filter(self):
        hits = BM25Retriever(CHUNKS).search('ngân hàng', filters={'section': 'a'}, top_k=3)
        self.assertEqual([h['id'] for h in hits], ['c3'])

    def test_bm25_drops_zero_score_chunks(self):
        self.assertEqual(BM25Retriever(CHUNKS).search('bitcoin', filters=None, top_k=3), [])

    def test_dense_maps_qdrant_hit_to_contract(self):
        store = FakeStore()
        hit = DenseRetriever(store, FakeEmbedder()).search('q', filters={'section': 'a'}, top_k=4)[0]
        self.assertEqual(store.args, ('C', 4, {'section': 'a'}))
        self.assertEqual((hit['id'], hit['page'], hit['source_id'], hit['method']), ('c9', 7, 'doc', 'dense'))

    def test_rrf_fuses_rankings(self):
        hybrid = HybridRRF([FixedRetriever(['a', 'b', 'c']), FixedRetriever(['b', 'c', 'a'])], k=60, candidates=3)
        self.assertEqual([h['id'] for h in hybrid.search('q', filters=None, top_k=3)], ['b', 'a', 'c'])

    def test_rerank_orders_by_llm_scores_and_ignores_unknown_ids(self):
        llm = FakeLLM('{"scores": {"a": 1, "b": 9, "zzz": 10}}')
        hits = LLMRerank(FixedRetriever(['a', 'b', 'c']), llm, candidates=3).search('q', filters=None, top_k=2)
        self.assertEqual([h['id'] for h in hits], ['b', 'a'])
        self.assertEqual(hits[0]['method'], 'fixed+rerank')

    def test_rerank_falls_back_to_base_order_on_bad_json(self):
        hits = LLMRerank(FixedRetriever(['a', 'b']), FakeLLM('not json'), candidates=2).search('q', filters=None, top_k=2)
        self.assertEqual([h['id'] for h in hits], ['a', 'b'])
        self.assertTrue(hits[0]['rerank_error'])


if __name__ == '__main__':
    unittest.main()
