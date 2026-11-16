"""Offline checks for the LLM retry/cache helpers (no network)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.day7.llm import DiskCache, ProviderError, call_with_retry


class RetryTests(unittest.TestCase):
    def test_retries_rate_limit_then_succeeds(self):
        calls, waits = [], []
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError('429 RESOURCE_EXHAUSTED. Please retry in 2.5s.')
            return 'ok'
        self.assertEqual(call_with_retry(fn, attempts=6, sleep=waits.append), 'ok')
        self.assertEqual(len(calls), 3)
        self.assertEqual(waits, [3.5, 3.5])

    def test_raises_provider_error_after_limit(self):
        def fn(): raise RuntimeError('503 UNAVAILABLE')
        with self.assertRaises(ProviderError):
            call_with_retry(fn, attempts=3, sleep=lambda s: None)

    def test_non_retryable_error_raises_immediately(self):
        calls = []
        def fn(): calls.append(1); raise RuntimeError('400 INVALID_ARGUMENT')
        with self.assertRaises(ProviderError):
            call_with_retry(fn, attempts=5, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)


class CacheTests(unittest.TestCase):
    def test_cache_round_trip_by_request(self):
        with tempfile.TemporaryDirectory() as d:
            cache = DiskCache(Path(d))
            req = {'model': 'm', 'messages': ['xin chào']}
            self.assertIsNone(cache.get(req))
            cache.put(req, {'text': 'chào'})
            self.assertEqual(DiskCache(Path(d)).get(req), {'text': 'chào'})
            self.assertIsNone(cache.get({'model': 'm', 'messages': ['khác']}))


if __name__ == '__main__':
    unittest.main()
