"""Offline checks for converting Day 6 Qdrant payloads to the Day 7 chunk contract."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.day7.export_day6 import payload_to_chunk


class ExportTests(unittest.TestCase):
    def test_payload_maps_to_day7_contract(self):
        payload = {'chunk_id': 'doc__C__p20__c001', 'document_id': 'doc', 'source': 'a.pdf', 'page': 20,
                   'text': 'Nội dung', 'strategy': 'C', 'section': 'chuong_1_kinh_te_vi_mo', 'year': 2024}
        chunk = payload_to_chunk(payload)
        self.assertEqual(chunk, {'id': 'doc__C__p20__c001', 'text': 'Nội dung', 'source_id': 'doc',
                                 'source': 'a.pdf', 'page': 20,
                                 'metadata': {'strategy': 'C', 'section': 'chuong_1_kinh_te_vi_mo', 'year': 2024}})

    def test_payload_without_text_is_rejected(self):
        with self.assertRaises(ValueError):
            payload_to_chunk({'chunk_id': 'x', 'document_id': 'd', 'source': 's', 'page': 1, 'text': ''})


if __name__ == '__main__':
    unittest.main()
