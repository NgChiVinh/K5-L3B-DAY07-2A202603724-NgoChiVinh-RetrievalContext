"""Print markdown tables from saved results (retrieval + agent) for the report.

Run: python -m src.day7.report_tables --split dev
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config


def frac(d):
    return f"{d['n']}/{d['N']} ({100 * d['rate']:.1f}%)" if d and d.get('N') else '0/0'


def retrieval_table(split: str) -> str:
    rows = ['| Phương pháp | Hit@1 | Evidence Hit@3 | MRR@10 | Latency TB / câu | LLM tokens (in/out) | Lỗi |', '|---|---|---|---|---|---|---|']
    for p in sorted((config.RESULTS_DIR / 'retrieval' / split).glob('*.json')):
        s = json.loads(p.read_text(encoding='utf-8'))['summary']
        rows.append(f"| {s['method']} | {frac(s['hit_at_1'])} | {frac(s['evidence_hit_at_3'])} | {s['mrr']:.3f} | "
                    f"{s['latency_s_mean']}s | {s['llm_input_tokens']}/{s['llm_output_tokens']} | "
                    f"{len(s['errors'])} (rerank lỗi {s.get('rerank_errors', 0)}) |")
    return '\n'.join(rows)


def agent_table(split: str) -> str:
    rows = ['| Config | Tools / top-k / budget / max search / neighbours | Tool use | Searches | Context Evidence Rate | '
            'Từ chối đúng (thiếu ev.) | Từ chối nhầm | Context TB (ký tự) | Latency TB | LLM calls / tokens in+out+think | Lỗi |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for p in sorted((config.RESULTS_DIR / 'agent' / split).glob('*/summary.json')):
        s = json.loads(p.read_text(encoding='utf-8'))
        c, u = s['config'], s['usage_total']
        rows.append(f"| {s['config_name']} | {'+'.join(c['tools'])} / {c['top_k']} / {c['budget_chars']} / {c['max_searches']} / "
                    f"{'có' if c['neighbors'] else 'không'} | {frac(s['tool_use'])} | {s['searches_distribution']} | "
                    f"{frac(s['context_evidence_rate'])} | {frac(s['abstained_unanswerable'])} | {frac(s['refused_answerable'])} | "
                    f"{s['mean_context_chars']} | {s['latency_s_mean']}s | {u.get('llm_calls', 0)} / "
                    f"{u.get('input_tokens', 0)}+{u.get('output_tokens', 0)}+{u.get('thinking_tokens', 0)} | {s['errors']} |")
    return '\n'.join(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['dev', 'test'], required=True)
    a = p.parse_args()
    if (config.RESULTS_DIR / 'retrieval' / a.split).exists():
        print(f'### Retrieval ({a.split}, query gốc + filter người dùng)\n\n{retrieval_table(a.split)}\n')
    print(f'### Agent ({a.split})\n\n{agent_table(a.split)}')


if __name__ == '__main__':
    main()
