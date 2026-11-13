"""OpenAI-compatible chat-completions player (OpenAI, OpenRouter, Ollama, LM Studio,
vLLM, Groq, Together...). Stdlib only (urllib) to keep deps small.

Ollama: base_url="http://localhost:11434/v1", api_key_env unset.
OpenRouter: base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY".
"""
from __future__ import annotations

import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from email.utils import parsedate_to_datetime

from ..protocol import SYSTEM_PROMPT, Request, extract_json, user_prompt
from .base import Player, Reply


class OpenAICompatPlayer(Player):
    def __init__(self, model: str, id: str | None = None,
                 base_url: str = "https://api.openai.com/v1", api_key_env: str | None = "OPENAI_API_KEY",
                 temperature: float = 0.2, max_tokens: int | None = None, json_mode: bool = True,
                 retries: int = 1, extra_body: dict | None = None, extra_headers: dict | None = None,
                 backoff_base: float = 0.5):
        self.model, self.id = model, id or model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.temperature, self.max_tokens, self.json_mode = temperature, max_tokens, json_mode
        self.retries, self.extra_body, self.extra_headers = retries, extra_body or {}, extra_headers or {}
        self.backoff_base = backoff_base
        self._rng = random.Random()

    def _post(self, body: dict, timeout: float, cancel: threading.Event | None = None) -> dict:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers, method="POST")
        with _urlopen(req, timeout=timeout, cancel=cancel) as r:
            return json.loads(r.read().decode())

    def act(self, req: Request) -> Reply:
        deadline = time.monotonic() + req.time_limit_s
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(req)}]
        usage: dict = {}
        last_err, raw = "", ""
        # Local to this call: a 400 from response_format must not flip the player
        # for every later request.
        json_mode = self.json_mode
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 1 or (req.cancel is not None and req.cancel.is_set()):
                return Reply(error=last_err or "timeout", raw=raw, usage=usage)
            body = {"model": self.model, "messages": msgs, "temperature": self.temperature,
                    **self.extra_body}
            if self.max_tokens:
                body["max_tokens"] = self.max_tokens
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            try:
                resp = self._post(body, timeout=remaining, cancel=req.cancel)
            except urllib.error.HTTPError as e:
                raw_body = e.read()
                _merge_usage(usage, _usage_from_body(raw_body))
                last_err = f"http {e.code}: {raw_body[:300]!r}"
                if e.code == 400 and json_mode:  # some servers reject response_format
                    json_mode = False
                    continue
                if _transient(e.code) and attempt < self.retries:
                    delay = _backoff_delay(attempt, e.headers.get("Retry-After"),
                                           self.backoff_base, self._rng)
                    if _wait(delay, deadline, req.cancel):
                        continue
                break  # 401 and other 4xx, or no time left to back off
            except Exception as e:  # timeout, connection refused...
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.retries and _wait(
                    _backoff_delay(attempt, None, self.backoff_base, self._rng),
                    deadline, req.cancel,
                ):
                    continue
                break
            _merge_usage(usage, resp.get("usage"))
            try:
                raw = resp["choices"][0]["message"]["content"] or ""
                return Reply(data=extract_json(raw, schema=req.output_schema), raw=raw[:4000], usage=usage)
            except (KeyError, IndexError, ValueError) as e:
                last_err = f"parse: {e}"
                # one repair turn: show the model its output and ask for JSON only
                msgs = msgs[:2] + [{"role": "assistant", "content": raw[:4000]},
                                   {"role": "user", "content": "That was not a single valid JSON "
                                    "object matching the schema. Reply with the JSON object only."}]
        return Reply(error=last_err, raw=raw[:4000], usage=usage)



def _transient(code: int) -> bool:
    """429 and 5xx are worth retrying. 401 and other 4xx are not."""
    return code == 429 or code >= 500


def _merge_usage(acc: dict, u: dict | None) -> None:
    for k, v in (u or {}).items():
        if isinstance(v, (int, float)):
            acc[k] = acc.get(k, 0) + v


def _usage_from_body(raw: bytes) -> dict:
    try:
        data = json.loads(raw.decode() or "{}")
    except (ValueError, UnicodeDecodeError):
        return {}
    return data.get("usage") or {} if isinstance(data, dict) else {}


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


def _backoff_delay(attempt: int, retry_after: str | None, base: float,
                   rng: random.Random) -> float:
    """Retry-After wins when present; otherwise exponential backoff with full jitter."""
    parsed = _parse_retry_after(retry_after)
    if parsed is not None:
        return min(30.0, parsed)
    return rng.uniform(0.0, min(30.0, base * (2 ** attempt)))


def _wait(delay: float, deadline: float, cancel: threading.Event | None) -> bool:
    """Sleep up to `delay`, but leave 1s for the next attempt. False = do not retry."""
    remaining = deadline - time.monotonic()
    if remaining <= 1:
        return False
    delay = min(max(0.0, delay), remaining - 1)
    if delay <= 0:
        return False
    end = time.monotonic() + delay
    while True:
        if cancel is not None and cancel.is_set():
            return False
        left = end - time.monotonic()
        if left <= 0:
            return True
        time.sleep(min(0.05, left))

@contextmanager
def _urlopen(req: urllib.request.Request, timeout: float, cancel: threading.Event | None):
    """urlopen that closes the HTTP connection when `cancel` is set.

    A socket timeout alone is not enough: urlopen's timeout is per socket
    operation, so a server that dribbles bytes can outlive the deadline, and
    an abandoned thread keeps the request open (and billable) after the
    referee has already scored a timeout.
    """
    holder: list = []

    def http_cls(*a, **k):
        c = http.client.HTTPConnection(*a, **k)
        holder.append(c)
        return c

    def https_cls(*a, **k):
        c = http.client.HTTPSConnection(*a, **k)
        holder.append(c)
        return c

    class HTTP(urllib.request.HTTPHandler):
        def http_open(self, r):
            return self.do_open(http_cls, r)

    class HTTPS(urllib.request.HTTPSHandler):
        def https_open(self, r):
            return self.do_open(https_cls, r)

    stop = threading.Event()

    def watch():
        while not stop.wait(0.05):
            if cancel is not None and cancel.is_set():
                break
        else:
            return
        for c in list(holder):
            try:
                c.close()
            except Exception:
                pass

    if cancel is not None:
        threading.Thread(target=watch, daemon=True).start()
    try:
        with urllib.request.build_opener(HTTP(), HTTPS()).open(req, timeout=timeout) as r:
            yield r
    finally:
        stop.set()
