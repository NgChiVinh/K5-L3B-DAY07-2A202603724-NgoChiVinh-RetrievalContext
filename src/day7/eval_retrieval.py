"""Benchmark retrieval methods on the original question (+ user-provided filter).

Run: python -m src.day7.eval_retrieval --split dev [--methods dense bm25 hybrid ...]
Writes results/retrieval/<split>/<method>.json with rankings (top-10), Hit@1, Evidence Hit@3, MRR, latency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config
from .benchmark import retrieval_metrics
from .configs import RETRIEVAL_METHODS, Context

TOP_N = 10


def run(split: str, methods: list[str]) -> dict[str, dict]:
    ctx = Context()
    questions = [q for q in json.loads(config.QUESTIONS_PATH.read_text(encoding='utf-8')) if q['split'] == split]
    out_dir = config.RESULTS_DIR / 'retrieval' / split
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in methods:
        retriever = ctx.retriever(name)
        rankings, traces, errors, latencies = {}, [], [], []
        for q in questions:  # only id/question/filter reach the retriever
            t0 = time.time()
            try:
                hits = retriever.search(q['question'], filters=q.get('filter'), top_k=TOP_N)
            except Exception as exc:
                errors.append({'query_id': q['id'], 'error': f'{type(exc).__name__}: {exc}'})
                hits = []
            latencies.append(time.time() - t0)
            rankings[q['id']] = hits
            traces.append({'query_id': q['id'], 'query': q['question'], 'filter': q.get('filter'),
                           'results': [{k: h.get(k) for k in ('rank', 'id', 'page', 'score', 'method', 'base_rank',
                                                            'component_ranks', 'rerank_error')} for h in hits]})
        m = retrieval_metrics(rankings, questions)
        usage = getattr(retriever, 'usage', [])
        m.update({'method': name, 'split': split, 'errors': errors,
                  'rerank_errors': sum(bool(t['results'] and t['results'][0].get('rerank_error')) for t in traces),
                  'llm_model': ctx.llm.model if 'rerank' in name else None,
                  'latency_s_mean': round(sum(latencies) / len(latencies), 3),
                  'llm_calls': len(usage), 'llm_input_tokens': sum(u.get('input_tokens', 0) for u in usage),
                  'llm_output_tokens': sum(u.get('output_tokens', 0) for u in usage)})
        (out_dir / f'{name}.json').write_text(json.dumps({'summary': m, 'traces': traces}, ensure_ascii=False, indent=1),
                                              encoding='utf-8')
        summary[name] = m
        print(f"{name:14s} Hit@1 {m['hit_at_1']['n']}/{m['hit_at_1']['N']}  EH@3 {m['evidence_hit_at_3']['n']}/"
              f"{m['evidence_hit_at_3']['N']}  MRR {m['mrr']:.3f}  lat {m['latency_s_mean']}s  errors {len(errors)} rerank_err {m['rerank_errors']}")
    return summary


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['dev', 'test'], required=True)
    p.add_argument('--methods', nargs='+', default=RETRIEVAL_METHODS, choices=RETRIEVAL_METHODS)
    a = p.parse_args()
    run(a.split, a.methods)


if __name__ == '__main__':
    main()
