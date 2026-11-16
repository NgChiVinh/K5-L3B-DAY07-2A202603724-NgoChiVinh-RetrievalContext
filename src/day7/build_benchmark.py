"""Build data/day7/questions.json: 20 Day 6 answerable questions + 4 unanswerable ones.

Run: python -m src.day7.build_benchmark
Day 6 split is kept (q01-q15 dev, q16-q20 test). Gold is converted to `gold_groups`;
q11 and q19 need several chunks under the Day 6 strategy-C chunking (see OVERRIDES).
Every gold alternative is checked to lie fully inside one chunk of the right page.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from . import config
from .benchmark import _alt_hit, normalize

DOC = 'wb-diem-lai-2024-08'


def alt(page: int, evidence: str) -> dict[str, Any]:
    return {'source_id': DOC, 'page': page, 'evidence': evidence}


# q11 asks for three barriers -> three required groups on p.50 (p.60 summary names all three).
P60 = alt(60, 'yêu cầu ký quỹ trước giao dịch, tỷ lệ cổ phiếu tối đa được phép sở hữu do giới hạn tỷ lệ sở hữu vốn của nhà đầu tư nước ngoài, công bằng trong tiếp cận thông tin')
OVERRIDES: dict[str, list[list[dict[str, Any]]]] = {
    'q11': [[alt(50, '(i) Ký quỹ và phong tỏa chứng khoán trước giao dịch'), P60],
            [alt(50, '(ii) Giới hạn tỷ lệ sở hữu vốn của nhà đầu tư nước ngoài'), P60],
            [alt(50, '(iii) Tiếp cận thông tin công bằng'), P60]],
    # q19: "37 ngàn tỷ" and "ba đợt IPO" sit in two different chunks of p.49.
    'q19': [[alt(49, 'bình quân đạt 37 ngàn tỷ đồng (1,5 tỷ đô la Mỹ) mỗi năm trong năm năm qua'),
             alt(40, 'bình quân chỉ đạt 37 ngàn tỷ đồng (1,5 tỷ đô la Mỹ) mỗi năm trên hai sàn giao dịch chứng khoán')],
            [alt(49, 'Trong năm 2023, chỉ có ba đợt IPO được thực hiện, huy động được 173 tỷ đồng')]],
    # Label QA (read against the PDF text, before any test run): one group per fact the question asks.
    'q09': [[alt(41, 'I#1: Công cụ (Instrument)')],
            [alt(42, 'I#2: Nhà đầu tư (Investor)')],   # same chunk also defines I#3, I#4
            [alt(42, 'I#5: Hạ tầng (Infrastructure)')]],  # same chunk also defines I#6
    # q20 asks for the total; only p.61 states it (p.60 quote covers the 20 tỷ BHXH part only).
    'q20': [[alt(61, 'riêng các thị trường vốn sẽ dành được khoảng 87 tỷ đô la Mỹ cho khu vực doanh nghiệp')]],
}

# Day 6 quote covers only part of the expected answer: add one required group per missing fact.
ADD_GROUPS: dict[str, list[list[dict[str, Any]]]] = {
    'q05': [[alt(14, 'đồng đô-la Mỹ vẫn ở mức cao (166 điểm cơ bản)'), alt(28, '(166 điểm cơ bản)')]],
    'q08': [[alt(39, 'nguồn lực bổ sung ước tính chiếm 7% GDP chỉ riêng cho các dự án cơ sở hạ tầng')]],
    'q14': [[alt(51, 'trái phiếu chính phủ chiếm đến 86% tài sản thuộc quản lý của BHXH')]],
}

UNANSWERABLE = [
    {'id': 'u01', 'split': 'dev', 'type': 'thieu_bang_chung',
     'question': 'Ngân hàng Thế giới dự báo tốc độ tăng trưởng GDP thực của Việt Nam năm 2027 là bao nhiêu phần trăm?',
     'absence_reason': 'Báo cáo chỉ dự báo đến năm 2026 (Bảng 0.1 trang 16, Bảng 1.1 trang 36: 6,1% năm 2024; 6,5% năm 2025–2026).',
     'absence_check': "Không chunk nào chứa '2027'; bảng dự báo dừng ở 2026."},
    {'id': 'u02', 'split': 'dev', 'type': 'thieu_bang_chung',
     'question': 'Trong năm 2023, các doanh nghiệp Việt Nam đã phát hành bao nhiêu tỷ đồng trái phiếu xanh?',
     'absence_reason': 'Báo cáo có phân tích trái phiếu doanh nghiệp nhưng không có số liệu trái phiếu xanh.',
     'absence_check': "Không chunk nào chứa 'trái phiếu xanh'; 'xanh' chỉ xuất hiện ở chú thích màu (tr.21), 'xanh hóa' (tr.37), quỹ xanh (tr.53)."},
    {'id': 'u03', 'split': 'test', 'type': 'thieu_bang_chung',
     'question': 'Giá vàng miếng trong nước đã tăng bao nhiêu phần trăm trong nửa đầu năm 2024?',
     'absence_reason': 'Báo cáo chỉ nêu nhu cầu mua vàng tăng 28,6% và vàng bị đội giá (tr.14, tr.28), không nêu % tăng giá vàng.',
     'absence_check': "Các chunk chứa 'vàng' (tr.14, 17, 28, 39) không có tỷ lệ tăng giá vàng; 28,6% là nhu cầu mua."},
    {'id': 'u04', 'split': 'test', 'type': 'thieu_bang_chung',
     'question': 'Khối lượng giao dịch bình quân mỗi phiên của hợp đồng tương lai chỉ số VN30 năm 2023 là bao nhiêu hợp đồng?',
     'absence_reason': 'Báo cáo chỉ nhắc chung về công cụ phái sinh (tr.42–43), không có số liệu giao dịch phái sinh.',
     'absence_check': "'phái sinh' chỉ ở tr.42–43 (định nghĩa); không chunk nào chứa 'VN30' kèm hợp đồng tương lai."},
]


def day6_groups(q: dict[str, Any]) -> list[list[dict[str, Any]]]:
    alts = [alt(q['gold_page'], q['evidence_quote'])]
    alts += [alt(p, e) for p, e in zip(q.get('alternative_gold_pages') or [], q.get('alternative_evidence_quotes') or [])]
    return [alts]


def build(day6: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for q in day6:
        groups = (OVERRIDES.get(q['id']) or day6_groups(q)) + ADD_GROUPS.get(q['id'], [])
        for group in groups:  # keep only alternatives that lie inside a chunk; fail if a group is empty
            group[:] = [a for a in group if any(_alt_hit(c, a) for c in chunks)]
            if not group:
                raise ValueError(f"{q['id']}: gold group không nằm trọn trong chunk nào")
        out.append({'id': q['id'], 'split': q['split'], 'type': q['type'], 'question': q['question'],
                    'answerable': True, 'expected_behavior': 'search_and_answer',
                    'expected_answer': q['expected_answer'], 'filter': q.get('filter'),
                    'gold_groups': groups, 'gold': [a for g in groups for a in g]})
    for u in UNANSWERABLE:
        out.append({**u, 'answerable': False, 'expected_behavior': 'search_and_abstain', 'filter': None})
        if '2027' in u['question'] and any('2027' in normalize(c['text']) for c in chunks):
            raise ValueError('u01 không còn thiếu evidence')
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    chunks = json.loads(config.CHUNKS_PATH.read_text(encoding='utf-8'))
    day6 = json.loads((config.DATA_DIR / 'day6_questions.json').read_text(encoding='utf-8'))['questions']
    questions = build(day6, chunks)
    config.QUESTIONS_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    cases = [{'id': q['id'], 'split': q['split'], 'answerable': q['answerable']} for q in questions]
    (config.DATA_DIR / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for split in ('dev', 'test'):
        qs = [q for q in questions if q['split'] == split]
        print(split, len(qs), 'câu:', sum(q['answerable'] for q in qs), 'có đáp án,', sum(not q['answerable'] for q in qs), 'thiếu evidence')
    print('multi-group:', [q['id'] for q in questions if len(q.get('gold_groups') or []) > 1])


if __name__ == '__main__':
    main()
