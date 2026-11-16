"""Offline checks for evidence matching and retrieval metrics (no gold reaches the agent)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.day7.benchmark import context_has_evidence, evidence_hit, normalize, retrieval_metrics


def chunk(cid, text, page=20, source_id='doc'):
    return {'id': cid, 'text': text, 'source_id': source_id, 'source': 'a.pdf', 'page': page, 'metadata': {}}


def question(groups, qid='q1', answerable=True):
    return {'id': qid, 'split': 'dev', 'question': '?', 'answerable': answerable,
            'gold_groups': groups} if answerable else {'id': qid, 'split': 'dev', 'question': '?', 'answerable': False}


G1 = [{'source_id': 'doc', 'page': 20, 'evidence': 'tăng trưởng 7% so cùng kỳ'}]
G2 = [{'source_id': 'doc', 'page': 21, 'evidence': 'dịch vụ tăng 7,4%'}]


class BenchmarkTests(unittest.TestCase):
    def test_normalize_collapses_whitespace_and_case(self):
        self.assertEqual(normalize('Tăng\n  trưởng\t7%'), 'tăng trưởng 7%')

    def test_quote_with_line_break_in_chunk_still_hits(self):
        self.assertTrue(evidence_hit(chunk('c1', 'Sản lượng Tăng trưởng\n7% so cùng kỳ năm trước'), question([G1])))

    def test_wrong_page_misses(self):
        self.assertFalse(evidence_hit(chunk('c1', 'tăng trưởng 7% so cùng kỳ', page=13), question([G1])))

    def test_alternative_page_in_group_hits(self):
        alt = G1 + [{'source_id': 'doc', 'page': 13, 'evidence': 'tăng 7% so với cùng kỳ'}]
        self.assertTrue(evidence_hit(chunk('c1', 'xx tăng 7% so với cùng kỳ', page=13), question([alt])))

    def test_two_group_question_needs_both_groups(self):
        q = question([G1, G2])
        a, b = chunk('a', 'tăng trưởng 7% so cùng kỳ'), chunk('b', 'dịch vụ tăng 7,4%', page=21)
        self.assertFalse(context_has_evidence([a], q))
        self.assertTrue(context_has_evidence([a, b], q))

    def test_metrics_skip_unanswerable_and_use_cutoffs(self):
        qs = [question([G1], 'q1'), question([G1], 'q2'), question(None, 'q3', answerable=False)]
        good, other = chunk('g', 'tăng trưởng 7% so cùng kỳ'), chunk('o', 'không liên quan')
        rankings = {'q1': [good, other], 'q2': [other, other, good, good], 'q3': [other]}
        m = retrieval_metrics(rankings, qs)
        self.assertEqual((m['hit_at_1']['n'], m['hit_at_1']['N']), (1, 2))
        self.assertEqual((m['evidence_hit_at_3']['n'], m['evidence_hit_at_3']['N']), (2, 2))
        self.assertAlmostEqual(m['mrr'], (1 + 1 / 3) / 2)


if __name__ == '__main__':
    unittest.main()
