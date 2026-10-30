"""OpenAI-compatible chat-completions player (OpenAI, OpenRouter, Ollama, LM Studio,
vLLM, Groq, Together...). Stdlib only (urllib) to keep deps small.

Ollama: base_url="http://localhost:11434/v1", api_key_env unset.
OpenRouter: base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY".
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..protocol import SYSTEM_PROMPT, Request, extract_json, user_prompt
from .base import Player, Reply


class OpenAICompatPlayer(Player):
    def __init__(self, model: str, id: str | None = None,
                 base_url: str = "https://api.openai.com/v1", api_key_env: str | None = "OPENAI_API_KEY",
                 temperature: float = 0.2, max_tokens: int | None = None, json_mode: bool = True,
                 retries: int = 1, extra_body: dict | None = None, extra_headers: dict | None = None):
        self.model, self.id = model, id or model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.temperature, self.max_tokens, self.json_mode = temperature, max_tokens, json_mode
        self.retries, self.extra_body, self.extra_headers = retries, extra_body or {}, extra_headers or {}

    def _post(self, body: dict, timeout: float) -> dict:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def act(self, req: Request) -> Reply:
        deadline = time.monotonic() + req.time_limit_s
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(req)}]
        usage: dict = {}
        last_err, raw = "", ""
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                return Reply(error=last_err or "timeout", raw=raw, usage=usage)
            body = {"model": self.model, "messages": msgs, "temperature": self.temperature,
                    **self.extra_body}
            if self.max_tokens:
                body["max_tokens"] = self.max_tokens
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
            try:
                resp = self._post(body, timeout=remaining)
            except urllib.error.HTTPError as e:
                last_err = f"http {e.code}: {e.read()[:300]!r}"
                if e.code == 400 and self.json_mode:  # some servers reject response_format
                    self.json_mode = False
                continue
            except Exception as e:  # timeout, connection refused...
                last_err = f"{type(e).__name__}: {e}"
                continue
            for k, v in (resp.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
            try:
                raw = resp["choices"][0]["message"]["content"] or ""
                return Reply(data=extract_json(raw), raw=raw[:4000], usage=usage)
            except (KeyError, IndexError, ValueError) as e:
                last_err = f"parse: {e}"
                # one repair turn: show the model its output and ask for JSON only
                msgs = msgs[:2] + [{"role": "assistant", "content": raw[:4000]},
                                   {"role": "user", "content": "That was not a single valid JSON "
                                    "object matching the schema. Reply with the JSON object only."}]
        return Reply(error=last_err, raw=raw[:4000], usage=usage)
