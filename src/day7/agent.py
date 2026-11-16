"""Multi-tool agentic RAG loop with native function calling.

The model decides which search tool to call, with which query, and when to stop.
Context = every chunk shown to the model in tool results, deduplicated, limited by
`budget_chars` (sum of chunk text lengths). Optional neighbours add the adjacent chunk
(same page, index ±1). An answer is accepted only if it cites [chunk-id]s that are all in
that context; otherwise the run abstains. No gold/benchmark field enters this module.
"""
from __future__ import annotations

import re
import time
from typing import Any

from .prompts import ABSTAIN_MARKER, SYSTEM, tool_specs, user_message

ABSTAIN_TEXT = 'Không đủ bằng chứng để trả lời.'
_ID_RE = re.compile(r'__p(\d+)__c(\d+)$')


def agent_input(q: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """The only benchmark fields the agent may see: the question and the user-provided filter."""
    return q['question'], q.get('filter')


def _neighbors(hit: dict[str, Any], chunk_index: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    m = _ID_RE.search(hit['id'])
    if not m or not chunk_index:
        return []
    prefix, idx = hit['id'][:m.start()], int(m.group(2))
    wanted = {f'{prefix}__p{int(m.group(1)):02d}__c{i:03d}' for i in (idx - 1, idx + 1)}
    return [dict(c, method='neighbor', neighbor_of=hit['id']) for c in chunk_index if c['id'] in wanted]


def _format(c: dict[str, Any]) -> str:
    section = c.get('metadata', {}).get('section', '')
    return f"[{c['id']}] (trang {c['page']}, {section})\n{c['text']}"


def _citations(text: str) -> list[str]:
    """Chunk-id-shaped tokens inside [...] (Day 6 ids contain '__'); other brackets like [Hình 2.12] are prose."""
    out: list[str] = []
    for group in re.findall(r'\[([^\[\]]+)\]', text):
        for cid in re.split(r'[,;\s]+', group.strip()):
            if '__' in cid and cid not in out:
                out.append(cid)
    return out


def run_agent(question: str, llm, tools: dict[str, Any], *, filters: dict[str, Any] | None = None,
              max_searches: int = 4, top_k: int = 5, budget_chars: int = 4000, neighbors: bool = False,
              chunk_index: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not question.strip() or max_searches < 1 or top_k < 1 or budget_chars < 1 or not tools:
        raise ValueError('question/max_searches/top_k/budget/tools không hợp lệ')
    t0 = time.time()
    specs = tool_specs(list(tools))
    messages: list[dict[str, Any]] = [{'role': 'user', 'text': user_message(question, filters)}]
    events: list[dict[str, Any]] = []
    context: dict[str, dict[str, Any]] = {}
    used = 0
    searches = 0
    usage = {'llm_calls': 0, 'cached_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'thinking_tokens': 0}

    def finish(status: str, answer: str, citations: list[str], event: dict[str, Any]) -> dict[str, Any]:
        return {'status': status, 'answer': answer, 'citations': citations, 'events': events + [event],
                'context_ids': list(context), 'context': [_format(c) for c in context.values()],
                'context_chars': used, 'searches': searches, 'usage': usage,
                'latency_s': round(time.time() - t0, 3)}

    for _turn in range(max_searches + 2):
        resp = llm.chat(SYSTEM, messages, specs)
        usage['llm_calls'] += 1
        usage['cached_calls'] += int(bool(resp.get('cached')))
        for k in ('input_tokens', 'output_tokens', 'thinking_tokens'):
            usage[k] += int((resp.get('usage') or {}).get(k, 0) or 0)
        calls = resp.get('tool_calls') or []
        if not calls:
            text = (resp.get('text') or '').strip()
            if ABSTAIN_MARKER.casefold() in text.casefold()[:80]:
                return finish('abstain', text, [], {'event': 'abstain', 'model_text': text})
            cites = _citations(text)
            if text and cites and all(c in context for c in cites):
                return finish('answered', text, cites, {'event': 'answer', 'citations': cites})
            return finish('abstain', ABSTAIN_TEXT, [], {'event': 'invalid_or_missing_citations',
                                                       'model_text': text, 'cited': cites})
        messages.append({'role': 'assistant', 'text': resp.get('text') or '', 'tool_calls': calls, 'raw': resp.get('raw')})
        for call in calls:
            if searches >= max_searches:
                return finish('abstain', ABSTAIN_TEXT, [], {'event': 'search_limit', 'requested': call})
            searches += 1
            name, query = call['name'], str((call.get('args') or {}).get('query', '')).strip()
            try:
                if name not in tools:
                    raise KeyError(f'tool không tồn tại: {name}')
                if not query:
                    raise ValueError('query rỗng')
                results = tools[name].search(query, filters=filters, top_k=top_k)
            except Exception as exc:
                err = f'{type(exc).__name__}: {exc}'
                events.append({'event': 'tool_error', 'tool': name, 'query': query, 'filter': filters, 'error': err})
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'name': name, 'content': f'Lỗi công cụ: {err}'})
                continue
            shown, skipped = [], []
            for hit in results:
                for c in [hit] + (_neighbors(hit, chunk_index) if neighbors else []):
                    if c['id'] in context:
                        continue
                    if used + len(c['text']) > budget_chars:
                        skipped.append(c['id'])
                        continue
                    context[c['id']] = c
                    used += len(c['text'])
                    shown.append(c)
            events.append({'event': 'search', 'tool': name, 'query': query, 'filter': filters, 'top_k': top_k,
                           'results': [{k: h.get(k) for k in ('id', 'rank', 'score', 'page', 'method', 'base_rank',
                                                            'component_ranks', 'rerank_error')} for h in results],
                           'added_ids': [c['id'] for c in shown], 'skipped_over_budget': skipped,
                           'context_chars': used})
            body = '\n\n'.join(_format(c) for c in shown) or 'Không có đoạn mới (đã hiển thị trước đó, không có kết quả, hoặc hết ngân sách context).'
            if searches == max_searches:
                body += '\n\n(Đây là lượt tìm cuối cùng; hãy trả lời có citation hoặc từ chối.)'
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'name': name, 'content': body})
    return finish('abstain', ABSTAIN_TEXT, [], {'event': 'turn_limit'})
