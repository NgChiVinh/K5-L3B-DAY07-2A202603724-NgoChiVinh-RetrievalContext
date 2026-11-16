"""Offline checks for the advanced (multimodal) ingestion helpers."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adv.ingest import clean_page_text, merge_blocks, parse_vision, section_for_page, visual_chunks


class IngestTests(unittest.TestCase):
    def test_section_for_page_uses_pdf_page_ranges(self):
        self.assertEqual(section_for_page(16), 'tong_quan')
        self.assertEqual(section_for_page(40), 'chuong_1_thich_ung')
        self.assertEqual(section_for_page(120), 'phu_luc')

    def test_running_header_is_removed(self):
        raw = 'Con đường hướng tới tương lai bền vững  I  9  I Việt Nam cũng dễ bị tổn thương'
        self.assertEqual(clean_page_text(raw), 'Việt Nam cũng dễ bị tổn thương')
        raw2 = 'I  10  I  Việt Nam 2045 - Tăng trưởng xanh hơn 38 Chỉ số PMI'
        self.assertEqual(clean_page_text(raw2), '38 Chỉ số PMI')

    def test_merge_blocks_respects_max_size_and_order(self):
        blocks = ['a' * 400, 'b' * 400, 'c' * 400]
        self.assertEqual(merge_blocks(blocks, max_size=900), ['a' * 400 + '\n' + 'b' * 400, 'c' * 400])

    def test_parse_vision_accepts_fenced_json(self):
        out = parse_vision('```json\n{"elements": [{"kind": "chart", "label": "Hình 0.1", "title": "T", "region": "top", "content": "x"}]}\n```')
        self.assertEqual(out[0]['label'], 'Hình 0.1')

    def test_parse_vision_bad_output_returns_empty_list(self):
        self.assertEqual(parse_vision('không phải json'), [])

    def test_visual_chunk_text_and_ids(self):
        els = [{'kind': 'table', 'label': 'Bảng 1.1', 'title': 'Tăng trưởng', 'region': 'top', 'content': '| a | b |'}]
        c = visual_chunks(els, page=8, doc_id='doc', source='a.pdf')[0]
        self.assertEqual(c['id'], 'doc__p008__c101')
        self.assertEqual(c['metadata']['modality'], 'table')
        self.assertIn('Bảng 1.1', c['text'])
        self.assertIn('| a | b |', c['text'])
        self.assertEqual(c['metadata']['region'], 'top')


if __name__ == '__main__':
    unittest.main()
