"""Provider-neutral LLM client with native tool calling, retry on 429/5xx and a disk cache.

Neutral message format used by the agent:
  {'role': 'user', 'text': str}
  {'role': 'assistant', 'text': str, 'tool_calls': [{'id', 'name', 'args'}], 'raw': provider content}
  {'role': 'tool', 'tool_call_id': str, 'name': str, 'content': str}
Tools: [{'name', 'description', 'parameters': JSON schema}].
Errors are raised as ProviderError; there is no mock fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

RETRYABLE = ('429', 'RESOURCE_EXHAUSTED', '500', '502', '503', '504', 'UNAVAILABLE', 'DEADLINE_EXCEEDED',
             'INTERNAL', 'timed out', 'Timeout', 'Connection', 'overloaded')


class ProviderError(RuntimeError):
    pass


def call_with_retry(fn: Callable[[], Any], attempts: int = 6, sleep: Callable[[float], None] = time.sleep,
                    events: list[dict[str, Any]] | None = None) -> Any:
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            if not any(t in msg for t in RETRYABLE) or i == attempts - 1:
                raise ProviderError(msg[:500]) from exc
            m = re.search(r'retry in ([0-9.]+)s', msg, re.I)
            wait = float(m.group(1)) + 1 if m else min(60.0, 5.0 * 2 ** i)
            if events is not None:
                events.append({'attempt': i + 1, 'wait_s': round(wait, 1), 'error': msg[:200]})
            sleep(wait)


class DiskCache:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def _path(self, request: dict[str, Any]) -> Path:
        blob = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)
        return self.dir / f'{hashlib.sha256(blob.encode("utf-8")).hexdigest()}.json'

    def get(self, request: dict[str, Any]) -> Any:
        p = self._path(request)
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

    def put(self, request: dict[str, Any], value: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path(request).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


class LLMClient:
    """Real model client. provider: 'gemini' (google-genai) or 'openai'."""

    def __init__(self, provider: str, model: str, cache_dir: Path | None = None, temperature: float = 0.0,
                 thinking_level: str | None = 'low'):
        self.provider, self.model, self.temperature, self.thinking_level = provider, model, temperature, thinking_level
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self.retry_events: list[dict[str, Any]] = []
        if provider == 'gemini':
            from google import genai
            key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not key:
                raise ProviderError('GEMINI_API_KEY chưa được cấu hình trong .env')
            self._client = genai.Client(api_key=key)
        elif provider == 'openai':
            from openai import OpenAI
            if not os.getenv('OPENAI_API_KEY'):
                raise ProviderError('OPENAI_API_KEY chưa được cấu hình trong .env')
            self._client = OpenAI()
        else:
            raise ValueError(f'provider không hỗ trợ: {provider}')

    # ---- public API -------------------------------------------------------
    def complete(self, system: str, user: str) -> dict[str, Any]:
        return self.chat(system, [{'role': 'user', 'text': user}], tools=None)

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        request = {'provider': self.provider, 'model': self.model, 'temperature': self.temperature,
                   'thinking': self.thinking_level, 'system': system, 'messages': messages, 'tools': tools}
        if self.cache and (hit := self.cache.get(request)) is not None:
            return {**hit, 'cached': True, 'latency_s': 0.0}
        t0 = time.time()
        fn = self._gemini if self.provider == 'gemini' else self._openai
        out = call_with_retry(lambda: fn(system, messages, tools), events=self.retry_events)
        out['latency_s'] = round(time.time() - t0, 3)
        if self.cache:
            self.cache.put(request, out)
        return {**out, 'cached': False}

    # ---- Gemini -------------------------------------------------------------
    def _gemini(self, system, messages, tools):
        from google.genai import types
        contents: list[Any] = []
        for m in messages:
            if m['role'] == 'user':
                contents.append(types.Content(role='user', parts=[types.Part(text=m['text'])]))
            elif m['role'] == 'assistant':
                if m.get('raw'):
                    contents.append(types.Content.model_validate(m['raw']))
                else:
                    contents.append(types.Content(role='model', parts=[types.Part(text=m.get('text') or '')]))
            elif m['role'] == 'tool':
                part = types.Part.from_function_response(name=m['name'], response={'result': m['content']})
                if contents and contents[-1].role == 'user' and contents[-1].parts[0].function_response:
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role='user', parts=[part]))
        cfg: dict[str, Any] = {'system_instruction': system, 'temperature': self.temperature}
        if self.thinking_level:
            cfg['thinking_config'] = types.ThinkingConfig(thinking_level=self.thinking_level)
        if tools:
            cfg['tools'] = [types.Tool(function_declarations=[types.FunctionDeclaration(
                name=t['name'], description=t['description'], parameters_json_schema=t['parameters']) for t in tools])]
        resp = self._client.models.generate_content(model=self.model, contents=contents,
                                                    config=types.GenerateContentConfig(**cfg))
        if not resp.candidates or resp.candidates[0].content is None:
            raise ProviderError(f'Gemini trả về rỗng: {getattr(resp, "prompt_feedback", None)}')
        content = resp.candidates[0].content
        parts = content.parts or []
        text = ''.join(p.text for p in parts if p.text and not p.thought)
        calls = [{'id': f'call_{i}', 'name': p.function_call.name, 'args': dict(p.function_call.args or {})}
                 for i, p in enumerate(q for q in parts if q.function_call)]
        u = resp.usage_metadata
        usage = {'input_tokens': u.prompt_token_count or 0, 'output_tokens': u.candidates_token_count or 0,
                 'thinking_tokens': u.thoughts_token_count or 0}
        return {'text': text, 'tool_calls': calls, 'usage': usage,
                'raw': content.model_dump(mode='json', exclude_none=True),
                'finish_reason': str(resp.candidates[0].finish_reason)}

    # ---- OpenAI -------------------------------------------------------------
    def _openai(self, system, messages, tools):
        msgs: list[dict[str, Any]] = [{'role': 'system', 'content': system}]
        for m in messages:
            if m['role'] == 'user':
                msgs.append({'role': 'user', 'content': m['text']})
            elif m['role'] == 'assistant':
                msg: dict[str, Any] = {'role': 'assistant', 'content': m.get('text') or None}
                if m.get('tool_calls'):
                    msg['tool_calls'] = [{'id': c['id'], 'type': 'function', 'function': {
                        'name': c['name'], 'arguments': json.dumps(c['args'], ensure_ascii=False)}} for c in m['tool_calls']]
                msgs.append(msg)
            elif m['role'] == 'tool':
                msgs.append({'role': 'tool', 'tool_call_id': m['tool_call_id'], 'content': m['content']})
        kwargs: dict[str, Any] = {'model': self.model, 'messages': msgs, 'temperature': self.temperature}
        if tools:
            kwargs['tools'] = [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'],
                                                                 'parameters': t['parameters']}} for t in tools]
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        calls = [{'id': c.id, 'name': c.function.name, 'args': json.loads(c.function.arguments or '{}')}
                 for c in (choice.tool_calls or [])]
        usage = {'input_tokens': resp.usage.prompt_tokens, 'output_tokens': resp.usage.completion_tokens, 'thinking_tokens': 0}
        return {'text': choice.content or '', 'tool_calls': calls, 'usage': usage, 'raw': None,
                'finish_reason': resp.choices[0].finish_reason}
