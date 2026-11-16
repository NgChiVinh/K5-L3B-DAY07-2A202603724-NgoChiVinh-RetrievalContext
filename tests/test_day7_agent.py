"""Offline behavioural checks for the Day 7 agent loop with a scripted LLM (not a quality measure)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.day7.agent import agent_input, run_agent


def chunk(cid, text, page=49):
    return {'id': cid, 'text': text, 'source_id': 'doc', 'source': 'a.pdf', 'page': page, 'metadata': {'section': 's'}}


C1 = chunk('doc__C__p49__c001', 'Huy động vốn bình quân đạt 37 ngàn tỷ đồng.')
C2 = chunk('doc__C__p49__c002', 'Trong năm 2023, chỉ có ba đợt IPO.')


class Retriever:
    def __init__(self, results=None, error=None): self.results, self.error, self.calls = results or [], error, []
    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k))
        if self.error:
            raise self.error
        return [dict(c, score=0.9, rank=i, method='dense') for i, c in enumerate(self.results[:top_k], 1)]


def call(query, name='search_dense'):
    return {'text': '', 'tool_calls': [{'id': 'call_0', 'name': name, 'args': {'query': query}}],
            'usage': {'input_tokens': 10, 'output_tokens': 2}, 'raw': None}


def final(text):
    return {'text': text, 'tool_calls': [], 'usage': {'input_tokens': 20, 'output_tokens': 5}, 'raw': None}


class ScriptedLLM:
    def __init__(self, turns): self.turns, self.seen = list(turns), []
    def chat(self, system, messages, tools):
        self.seen.append([dict(m) for m in messages]); return self.turns.pop(0)


class AgentTests(unittest.TestCase):
    def test_search_then_cited_answer_is_answered(self):
        r = Retriever([C1])
        llm = ScriptedLLM([call('huy động vốn'), final(f'37 ngàn tỷ đồng [{C1["id"]}]')])
        out = run_agent('Huy động bao nhiêu?', llm, {'search_dense': r}, filters={'section': 's'}, top_k=2)
        self.assertEqual(out['status'], 'answered')
        self.assertEqual(out['citations'], [C1['id']])
        self.assertEqual(r.calls, [('huy động vốn', {'section': 's'}, 2)])
        self.assertEqual(out['events'][0]['event'], 'search')
        self.assertEqual(out['context_ids'], [C1['id']])
        self.assertIn(C1['text'], llm.seen[1][-1]['content'])

    def test_citation_outside_context_abstains(self):
        llm = ScriptedLLM([call('x'), final('Đáp án [doc__C__p99__c009]')])
        out = run_agent('?', llm, {'search_dense': Retriever([C1])})
        self.assertEqual(out['status'], 'abstain')
        self.assertEqual(out['events'][-1]['event'], 'invalid_or_missing_citations')

    def test_answer_without_search_abstains(self):
        out = run_agent('?', ScriptedLLM([final('Đáp án không nguồn')]), {'search_dense': Retriever([C1])})
        self.assertEqual(out['status'], 'abstain')

    def test_explicit_abstention_is_recorded(self):
        llm = ScriptedLLM([call('x'), final('KHÔNG ĐỦ BẰNG CHỨNG: tài liệu không nêu số liệu này.')])
        out = run_agent('?', llm, {'search_dense': Retriever([C1])})
        self.assertEqual((out['status'], out['citations']), ('abstain', []))
        self.assertEqual(out['events'][-1]['event'], 'abstain')

    def test_search_limit_stops_tool_loop(self):
        llm = ScriptedLLM([call('a'), call('b'), call('c')])
        out = run_agent('?', llm, {'search_dense': Retriever([C1])}, max_searches=2)
        self.assertEqual(out['status'], 'abstain')
        self.assertEqual(sum(e['event'] == 'search' for e in out['events']), 2)
        self.assertEqual(out['events'][-1]['event'], 'search_limit')

    def test_budget_limits_context(self):
        llm = ScriptedLLM([call('x'), final('KHÔNG ĐỦ BẰNG CHỨNG')])
        out = run_agent('?', llm, {'search_dense': Retriever([C1, C2])}, budget_chars=len(C1['text']) + 5)
        self.assertEqual(out['context_ids'], [C1['id']])

    def test_neighbors_add_adjacent_chunk_same_page(self):
        llm = ScriptedLLM([call('x'), final('KHÔNG ĐỦ BẰNG CHỨNG')])
        out = run_agent('?', llm, {'search_dense': Retriever([C1])}, neighbors=True, chunk_index=[C1, C2], top_k=1)
        self.assertEqual(out['context_ids'], [C1['id'], C2['id']])

    def test_tool_error_is_reported_to_model_and_traced(self):
        llm = ScriptedLLM([call('x'), final('KHÔNG ĐỦ BẰNG CHỨNG')])
        out = run_agent('?', llm, {'search_dense': Retriever(error=RuntimeError('qdrant down'))})
        self.assertEqual(out['events'][0]['event'], 'tool_error')
        self.assertIn('qdrant down', llm.seen[1][-1]['content'])

    def test_agent_input_exposes_only_question_and_filter(self):
        q = {'id': 'q1', 'question': 'Hỏi?', 'filter': {'section': 's'}, 'expected_answer': 'BÍ MẬT',
             'gold': [{'evidence': 'BÍ MẬT'}], 'gold_groups': [[{'evidence': 'BÍ MẬT'}]], 'answerable': True}
        self.assertEqual(agent_input(q), ('Hỏi?', {'section': 's'}))


if __name__ == '__main__':
    unittest.main()
