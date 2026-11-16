"""Run the real agent for one config on one split and save traces + automatic metrics.

Run: python -m src.day7.run_benchmark --config baseline_dense --split dev [--only q01 q02]
Output: results/agent/<split>/<config>/traces.json (agent_metrics.py compatible) and summary.json.
Only question + user filter reach the agent; gold is used afterwards for Context Evidence Rate.
Provider/vector-store failures are saved as status "error" (never replaced by fake output).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter

from . import config
from .agent import agent_input, run_agent
from .benchmark import context_has_evidence
from .configs import AGENT_CONFIGS, TOOL_TO_RETRIEVER, Context


def run(config_name: str, split: str, only: list[str] | None = None) -> dict:
    cfg = AGENT_CONFIGS[config_name]
    ctx = Context()
    by_id = {c['id']: c for c in ctx.chunks}
    questions = [q for q in json.loads(config.QUESTIONS_PATH.read_text(encoding='utf-8'))
                 if q['split'] == split and (not only or q['id'] in only)]
    tools = {t: ctx.retriever(TOOL_TO_RETRIEVER[t]) for t in cfg['tools']}
    out_dir = config.RESULTS_DIR / 'agent' / split / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    provider, model = config.llm_settings()
    traces = []
    for q in questions:
        question, filters = agent_input(q)
        t0 = time.time()
        try:
            r = run_agent(question, ctx.llm, tools, filters=filters, max_searches=cfg['max_searches'],
                          top_k=cfg['top_k'], budget_chars=cfg['budget_chars'], neighbors=cfg['neighbors'],
                          chunk_index=ctx.chunks)
        except Exception as exc:
            r = {'status': 'error', 'answer': '', 'citations': [], 'events': [{'event': 'error', 'error': f'{type(exc).__name__}: {exc}'}],
                 'context_ids': [], 'context': [], 'context_chars': 0, 'searches': 0, 'usage': {}, 'latency_s': round(time.time() - t0, 3)}
        ctx_chunks = [by_id[i] for i in r['context_ids'] if i in by_id]
        trace = {'query_id': q['id'], 'split': split, 'config': config_name, 'provider': provider, 'model': model,
                 'question': question, 'filter': filters, **r,
                 'auto_eval': {'answerable': q['answerable'],
                               'context_has_evidence': context_has_evidence(ctx_chunks, q) if q['answerable'] else None}}
        traces.append(trace)
        print(f"{q['id']} {r['status']:8s} searches={r['searches']} ctx_ev={trace['auto_eval']['context_has_evidence']} "
              f"lat={r['latency_s']}s  {r['answer'][:90]!r}", flush=True)
        (out_dir / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding='utf-8')
    summary = summarize(traces, cfg, config_name, split, provider, model)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'config'}, ensure_ascii=False))
    return summary


def summarize(traces, cfg, name, split, provider, model) -> dict:
    ok = [t for t in traces if t['status'] != 'error']
    ans = [t for t in traces if t['auto_eval']['answerable']]
    una = [t for t in traces if not t['auto_eval']['answerable']]
    usage = Counter()
    for t in traces:
        usage.update({k: v for k, v in (t.get('usage') or {}).items()})
    lat = [t['latency_s'] for t in ok]
    ratio = lambda n, d: {'n': n, 'N': d, 'rate': round(n / d, 4) if d else None}
    return {'config_name': name, 'config': cfg, 'split': split, 'provider': provider, 'model': model,
            'errors': len(traces) - len(ok),
            'tool_use': ratio(sum(any(e['event'] == 'search' for e in t['events']) for t in ok), len(ok)),
            'searches_distribution': dict(sorted(Counter(t['searches'] for t in ok).items())),
            'tool_calls_by_tool': dict(Counter(e['tool'] for t in ok for e in t['events'] if e['event'] == 'search')),
            'context_evidence_rate': ratio(sum(bool(t['auto_eval']['context_has_evidence']) for t in ans), len(ans)),
            'status_answerable': dict(Counter(t['status'] for t in ans)),
            'status_unanswerable': dict(Counter(t['status'] for t in una)),
            'abstained_unanswerable': ratio(sum(t['status'] == 'abstain' for t in una), len(una)),
            'refused_answerable': ratio(sum(t['status'] == 'abstain' for t in ans), len(ans)),
            'mean_context_chars': round(statistics.mean(t['context_chars'] for t in ok), 1) if ok else None,
            'latency_s_mean': round(statistics.mean(lat), 2) if lat else None,
            'latency_s_total': round(sum(lat), 1), 'usage_total': dict(usage),
            'cost_usd': 0.0, 'cost_note': 'Gemini API free tier (không trả phí); token ghi ở usage_total'}


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, choices=list(AGENT_CONFIGS))
    p.add_argument('--split', required=True, choices=['dev', 'test'])
    p.add_argument('--only', nargs='*')
    a = p.parse_args()
    if a.split == 'test':
        sel = config.RESULTS_DIR / 'selected_config.json'
        if not sel.exists():
            raise SystemExit('Chưa chốt cấu hình: tạo results/selected_config.json từ kết quả dev trước khi chạy test.')
    run(a.config, a.split, a.only)


if __name__ == '__main__':
    main()
