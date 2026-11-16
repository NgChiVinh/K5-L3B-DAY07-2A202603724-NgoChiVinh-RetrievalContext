"""Advanced challenge: multimodal ingestion of "Việt Nam 2045 – Tăng trưởng Xanh hơn" (160 pages).

Steps (python -m src.adv.ingest --step <name>):
  render  – PDF pages -> adv_store/pages/pNNN.png (100 dpi) for vision + human verification
  vision  – Gemini vision lists every table/chart/diagram/map/image on a page as JSON
            (tables transcribed to markdown, charts: axes/units/legend/values/trends)
  chunks  – text chunks (PyMuPDF blocks, header removed, merged <=1000 chars) + one chunk per
            visual element, written to data/adv/chunks.json
  index   – embed with gemini-embedding-001 (3072d) into Qdrant adv_store/qdrant
Every chunk keeps page (1-based PDF page), printed page label, section, modality, and for
visuals the element label (e.g. "Hình 1.4"), region on the page and the page image path.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from src.day7 import config as day7_config

REPO = day7_config.REPO
STORE = REPO / 'adv_store'
PDF = STORE / 'greener_vi.pdf'
PAGES_DIR = STORE / 'pages'
QDRANT_PATH = STORE / 'qdrant'
DATA = REPO / 'data' / 'adv'
VISION_PATH = DATA / 'vision_pages.json'
CHUNKS_PATH = DATA / 'chunks.json'
DOC_ID = 'wb-vn2045-greener-vi'
SOURCE = ('World Bank (2025). Việt Nam 2045 – Tăng trưởng Xanh hơn: Con đường hướng tới tương lai bền vững. '
          'CC BY-NC 3.0 IGO. https://openknowledge.worldbank.org/entities/publication/ff18e434-f1fa-4332-bbf3-c25d9eda49c3')
PAGE_OFFSET = 32  # printed arabic page N = PDF page N + 32
SECTIONS = [(1, 11, 'tien_van_ban'), (12, 33, 'tong_quan'), (34, 36, 'gioi_thieu'), (37, 62, 'chuong_1_thich_ung'),
            (63, 92, 'chuong_2_giam_phat_thai'), (93, 119, 'chuong_3_kinh_te_bien'), (120, 150, 'phu_luc'),
            (151, 160, 'tai_lieu_tham_khao')]
SKIP_PAGES = {2, 159}  # page-number only / blank

_HEADER_RE = re.compile(r'^\s*(Con đường hướng tới tương lai bền vững\s+I\s+[0-9ivxlc]+\s+I\s*'
                        r'|I\s+[0-9ivxlc]+\s+I\s+Việt Nam 2045 - Tăng trưởng xanh hơn\s*)', re.I)


def section_for_page(page: int) -> str:
    return next(name for lo, hi, name in SECTIONS if lo <= page <= hi)


def clean_page_text(text: str) -> str:
    return _HEADER_RE.sub('', text, count=1).strip()


def merge_blocks(blocks: list[str], max_size: int = 1000) -> list[str]:
    out: list[str] = []
    cur = ''
    for b in (x.strip() for x in blocks):
        if not b:
            continue
        if cur and len(cur) + 1 + len(b) > max_size:
            out.append(cur)
            cur = b
        else:
            cur = f'{cur}\n{b}' if cur else b
        while len(cur) > max_size:  # very long block: split on sentence end near the limit
            cut = cur.rfind('. ', 0, max_size)
            cut = cut + 1 if cut > max_size // 2 else max_size
            out.append(cur[:cut].strip())
            cur = cur[cut:].strip()
    if cur:
        out.append(cur)
    return out


def parse_vision(text: str) -> list[dict[str, Any]]:
    m = re.search(r'\{.*\}', text or '', re.S)
    if not m:
        return []
    try:
        els = json.loads(m.group(0)).get('elements', [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [e for e in els if isinstance(e, dict) and e.get('content')]


def printed_label(page: int) -> str:
    return str(page - PAGE_OFFSET) if page > PAGE_OFFSET else f'roman-{page}'


def visual_chunks(elements: list[dict[str, Any]], page: int, doc_id: str = DOC_ID, source: str = 'greener_vi.pdf') -> list[dict[str, Any]]:
    chunks = []
    for i, e in enumerate(elements, 1):
        kind = str(e.get('kind', 'image')).lower()
        label, title = (e.get('label') or '').strip(), (e.get('title') or '').strip()
        head = f"[{kind.upper()}] {label}{': ' if label and title else ''}{title}".strip()
        text = f"{head}\n{e['content'].strip()}\n(Trang PDF {page}, vùng: {e.get('region', 'không rõ')})"
        chunks.append({'id': f'{doc_id}__p{page:03d}__c{100 + i:03d}', 'text': text, 'source_id': doc_id, 'source': source,
                       'page': page, 'metadata': {'modality': kind, 'label': label, 'title': title,
                                                  'region': e.get('region'), 'section': section_for_page(page),
                                                  'printed_page': printed_label(page),
                                                  'page_image': f'adv_store/pages/p{page:03d}.png',
                                                  'extracted_by': 'gemini-vision'}})
    return chunks


VISION_SYSTEM = """Bạn là công cụ trích xuất nội dung trực quan từ ảnh một trang báo cáo tiếng Việt của Ngân hàng Thế giới.
Liệt kê MỌI bảng, biểu đồ, sơ đồ, bản đồ, ảnh, infographic có thông tin trên trang (bỏ qua logo, đường kẻ, ảnh trang trí, đoạn văn thường).
Với mỗi phần tử trả về:
- kind: "table" | "chart" | "diagram" | "map" | "image"
- label: số hiệu in trên trang (vd "Hình 1.4", "Bảng 2.1", "Hộp 3") hoặc "" nếu không có
- title: tiêu đề in trên trang
- region: vị trí trên trang ("top", "middle", "bottom", "full", "top-left"...)
- content: tiếng Việt, đầy đủ để trả lời câu hỏi mà không cần nhìn ảnh:
  * bảng: chép LẠI TOÀN BỘ bảng dạng markdown, giữ nguyên tiêu đề cột/hàng, đơn vị, số liệu, chú thích;
  * biểu đồ: loại biểu đồ, trục và đơn vị, chú giải (từng chuỗi), mọi nhãn số liệu in trên hình;
    nếu không có nhãn thì ước lượng giá trị đầu/cuối/đỉnh từng chuỗi và ghi "≈"; mô tả xu hướng, chuỗi cao/thấp nhất, nguồn;
  * sơ đồ/bản đồ/ảnh: các thành phần, chữ trên hình, quan hệ/mũi tên, màu sắc chú giải, nguồn.
VỚI MỖI BIỂU ĐỒ BẮT BUỘC có dòng "Giá trị ước lượng:" liệt kê từng chuỗi với giá trị tại điểm đầu, điểm cuối và các mốc trục hoành
(vd "Tiêu cực: 2030 ≈ -8,5%; 2040 ≈ -12%; 2050 ≈ -15,7%"); với biểu đồ cột/thanh: giá trị từng cột theo từng nhóm.
Chỉ ghi những gì nhìn thấy trên ảnh, không suy đoán thêm. Trả về đúng JSON: {"elements": [...]} ; nếu không có phần tử nào: {"elements": []}."""


def step_render() -> None:
    import pymupdf
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    for i, page in enumerate(doc, 1):
        out = PAGES_DIR / f'p{i:03d}.png'
        if not out.exists():
            page.get_pixmap(dpi=100).save(out)
    print(f'{doc.page_count} pages rendered to {PAGES_DIR}')


def step_vision(pages: list[int] | None, model: str) -> None:
    from src.day7.llm import LLMClient, ProviderError
    day7_config.load_env()
    llm = LLMClient('gemini', model, cache_dir=day7_config.LLM_CACHE_DIR)
    DATA.mkdir(parents=True, exist_ok=True)
    done = json.loads(VISION_PATH.read_text(encoding='utf-8')) if VISION_PATH.exists() else {}
    import pymupdf
    n = pymupdf.open(PDF).page_count
    for p in pages or range(1, n + 1):
        if p in SKIP_PAGES or (str(p) in done and done[str(p)].get('status') == 'ok'):
            continue
        try:
            r = llm.vision(VISION_SYSTEM, f'Trang PDF số {p}. Trích xuất các phần tử trực quan.', PAGES_DIR / f'p{p:03d}.png')
            els = parse_vision(r['text'])
            done[str(p)] = {'status': 'ok', 'model': model, 'elements': els, 'raw_ok': bool(els) or '"elements"' in r['text'],
                            'usage': r['usage'], 'latency_s': r['latency_s']}
        except ProviderError as exc:
            done[str(p)] = {'status': 'error', 'model': model, 'error': str(exc)[:300]}
            print(f'p{p}: ERROR {str(exc)[:120]}', flush=True)
            if 'quota exhausted' in str(exc):
                break
        VISION_PATH.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"p{p}: {done[str(p)].get('status')} {len(done[str(p)].get('elements', []))} elements", flush=True)


def step_chunks() -> None:
    import pymupdf
    doc = pymupdf.open(PDF)
    vision = json.loads(VISION_PATH.read_text(encoding='utf-8')) if VISION_PATH.exists() else {}
    chunks: list[dict[str, Any]] = []
    stats = {'text': 0, 'visual': 0, 'pages_indexed': 0, 'pages_vision_missing': []}
    for i, page in enumerate(doc, 1):
        if i in SKIP_PAGES:
            continue
        blocks = [b[4] for b in page.get_text('blocks') if b[6] == 0]
        if blocks:
            blocks[0] = clean_page_text(blocks[0])
        for j, text in enumerate(merge_blocks(blocks), 1):
            if len(text) < 30:
                continue
            chunks.append({'id': f'{DOC_ID}__p{i:03d}__c{j:03d}', 'text': text, 'source_id': DOC_ID, 'source': 'greener_vi.pdf',
                           'page': i, 'metadata': {'modality': 'text', 'section': section_for_page(i),
                                                   'printed_page': printed_label(i), 'extracted_by': 'pymupdf'}})
            stats['text'] += 1
        v = vision.get(str(i))
        if v and v.get('status') == 'ok':
            vc = visual_chunks(v['elements'], i)
            chunks.extend(vc)
            stats['visual'] += len(vc)
        else:
            stats['pages_vision_missing'].append(i)
        stats['pages_indexed'] += 1
    DATA.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (DATA / 'index_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False))


def step_index() -> None:
    from src.day6.models import Chunk
    from src.day6.store import QdrantVectorStore
    from src.day6.embeddings import MyEmbedder
    day7_config.load_env()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding='utf-8'))
    emb = MyEmbedder()
    vecs = emb.embed_documents([c['text'] for c in chunks])
    store = QdrantVectorStore(QDRANT_PATH, dimensions=emb.dimensions)
    objs = [Chunk(chunk_id=c['id'], document_id=c['source_id'], source=c['source'], page=c['page'], text=c['text'],
                  metadata=c['metadata']) for c in chunks]
    store.upsert('ADV', objs, vecs)
    print(f'indexed {store.count("ADV")} chunks, dim {emb.dimensions}, model {emb.model_name}, 429 events {len(emb.rate_limit_events)}')
    store.close()


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--step', required=True, choices=['render', 'vision', 'chunks', 'index'])
    p.add_argument('--pages', nargs='*', type=int)
    p.add_argument('--model', default='gemini-3.1-flash-lite')
    a = p.parse_args()
    {'render': step_render, 'chunks': step_chunks, 'index': step_index}.get(a.step, lambda: step_vision(a.pages, a.model))()


if __name__ == '__main__':
    main()
