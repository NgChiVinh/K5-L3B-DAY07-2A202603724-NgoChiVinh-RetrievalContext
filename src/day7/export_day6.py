"""Export Day 6 Qdrant chunks (strategy C) to data/day7/chunks.json (Day 7 contract).

Run: python -m src.day7.export_day6
"""
from __future__ import annotations

import json
import sys
from typing import Any

from . import config

CORE = ('chunk_id', 'document_id', 'source', 'page', 'text')


def payload_to_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    if not all(payload.get(k) not in (None, '') for k in CORE):
        raise ValueError(f'payload thiếu trường bắt buộc: {payload.get("chunk_id")}')
    return {'id': payload['chunk_id'], 'text': payload['text'], 'source_id': payload['document_id'],
            'source': payload['source'], 'page': int(payload['page']),
            'metadata': {k: v for k, v in payload.items() if k not in CORE}}


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(config.QDRANT_PATH))
    coll = f'day6_strategy_{config.COLLECTION_STRATEGY}'
    records, offset = [], None
    while True:
        batch, offset = client.scroll(coll, limit=256, offset=offset, with_payload=True, with_vectors=False)
        records.extend(batch)
        if offset is None:
            break
    client.close()
    chunks = sorted((payload_to_chunk(dict(r.payload)) for r in records), key=lambda c: c['id'])
    if len({c['id'] for c in chunks}) != len(chunks):
        raise ValueError('chunk id bị trùng')
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    pages = sorted({c['page'] for c in chunks})
    print(f'{coll}: {len(chunks)} chunks, pages {pages[0]}-{pages[-1]} ({len(pages)} pages) -> {config.CHUNKS_PATH}')


if __name__ == '__main__':
    main()
