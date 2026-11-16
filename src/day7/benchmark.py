"""Gold evidence matching and retrieval metrics. Used only by evaluation, never by the agent.

Question schema (data/day7/questions.json):
  answerable=True  -> gold_groups: list of groups; every group must be covered, any alternative
                      {source_id, page, evidence} inside a group suffices.
  answerable=False -> expected_behavior='search_and_abstain', absence_reason, absence_check.
A gold alternative matches a chunk when source_id and page are equal and the normalized
evidence string is a substring of the normalized chunk text.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text).casefold()).strip()


def _alt_hit(chunk: dict[str, Any], alt: dict[str, Any]) -> bool:
    return (chunk['source_id'] == alt['source_id'] and int(chunk['page']) == int(alt['page'])
            and normalize(alt['evidence']) in normalize(chunk['text']))


def context_has_evidence(chunks: list[dict[str, Any]], q: dict[str, Any]) -> bool:
    """True when every gold group is covered by at least one chunk of the list."""
    groups = q.get('gold_groups') or []
    return bool(groups) and all(any(_alt_hit(c, alt) for c in chunks for alt in group) for group in groups)


def evidence_hit(chunk: dict[str, Any], q: dict[str, Any]) -> bool:
    return context_has_evidence([chunk], q)


def _ratio(n: int, total: int) -> dict[str, Any]:
    return {'n': n, 'N': total, 'rate': round(n / total, 4) if total else None}


def retrieval_metrics(rankings: dict[str, list[dict[str, Any]]], questions: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [q for q in questions if q.get('answerable', True)]
    h1 = h3 = 0
    rr = 0.0
    per_question = {}
    for q in answerable:
        ranked = rankings.get(q['id'], [])
        a, b = context_has_evidence(ranked[:1], q), context_has_evidence(ranked[:3], q)
        first = next((i for i in range(1, len(ranked) + 1) if context_has_evidence(ranked[:i], q)), None)
        h1, h3 = h1 + a, h3 + b
        rr += 1 / first if first else 0.0
        per_question[q['id']] = {'hit_at_1': a, 'evidence_hit_at_3': b, 'first_full_rank': first}
    return {'hit_at_1': _ratio(h1, len(answerable)), 'evidence_hit_at_3': _ratio(h3, len(answerable)),
            'mrr': rr / len(answerable) if answerable else None, 'per_question': per_question}
