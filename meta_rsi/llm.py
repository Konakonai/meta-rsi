"""Minimal, zero-dependency LLM backends with token and latency accounting.

Supported backends:
  - openai-compat : POST {base_url}/chat/completions (vLLM on Colab G4, OpenAI, etc.)
  - gemini-cli    : local `gemini-api generate` CLI (uses OS keyring auth automatically)
  - gemini-rest   : POST https://generativelanguage.googleapis.com/v1/interactions
                    (uses GEMINI_API_KEY env var; ideal for Colab VMs without the CLI)
  - mock          : deterministic dummy client for unit tests
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    raw_error: str | None = None


@dataclass
class UsageTracker:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    latency_s: float = 0.0
    errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, r: LLMResponse) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += r.input_tokens
            self.output_tokens += r.output_tokens
            self.thought_tokens += r.thought_tokens
            self.latency_s += r.latency_s
            if r.raw_error:
                self.errors += 1

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "thought_tokens": self.thought_tokens,
                "latency_s": round(self.latency_s, 3),
                "errors": self.errors,
            }


def _extract_interaction_text(payload: dict) -> str:
    parts: list[str] = []
    for step in payload.get("steps") or []:
        if step.get("type") == "model_output":
            for item in step.get("content") or []:
                if item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
    if parts:
        return "".join(parts)
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    return ""


class LLMClient:
    def __init__(
        self,
        backend: str,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        timeout: float = 180.0,
        enable_thinking: bool = True,
        thinking_level: str | None = None,
        mock_fn: Callable[[str, str], str] | None = None,
    ):
        self.backend = backend
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.thinking_level = thinking_level
        self.mock_fn = mock_fn
        self.usage = UsageTracker()

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 4,
    ) -> LLMResponse:
        temp = self.temperature if temperature is None else temperature
        mtok = self.max_tokens if max_tokens is None else max_tokens
        t0 = time.time()
        if self.backend == "mock":
            text = self.mock_fn(system, user) if self.mock_fn else "```python\nprint(0)\n```"
            r = LLMResponse(text=text, latency_s=time.time() - t0, model=self.model)
            self.usage.record(r)
            return r
        if self.backend == "openai-compat":
            r = self._call_openai_compat(system, user, temp, mtok, retries)
        elif self.backend == "gemini-cli":
            r = self._call_gemini_cli(system, user, temp, mtok, retries)
        elif self.backend == "gemini-rest":
            r = self._call_gemini_rest(system, user, temp, mtok, retries)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        r.latency_s = time.time() - t0
        self.usage.record(r)
        return r

    def _call_openai_compat(self, system: str, user: str, temp: float, mtok: int, retries: int) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": mtok,
        }
        # Pass vLLM chat_template_kwargs for Gemma 4 / Qwen thinking toggle.
        payload["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        last_err = ""
        for i in range(retries):
            req = urllib.request.Request(url, body, headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                msg = data["choices"][0]["message"]
                text = msg.get("content") or ""
                if not text and msg.get("reasoning_content"):
                    text = msg["reasoning_content"]
                u = data.get("usage") or {}
                return LLMResponse(
                    text=text,
                    input_tokens=int(u.get("prompt_tokens") or 0),
                    output_tokens=int(u.get("completion_tokens") or 0),
                    model=self.model,
                )
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
                if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                    time.sleep(min(2 ** i, 16))
                    continue
                return LLMResponse(text="", model=self.model, raw_error=last_err)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
                last_err = str(e)
                if i < retries - 1:
                    time.sleep(min(2 ** i, 16))
                    continue
                return LLMResponse(text="", model=self.model, raw_error=last_err)
        return LLMResponse(text="", model=self.model, raw_error=last_err)

    def _call_gemini_cli(self, system: str, user: str, temp: float, mtok: int, retries: int) -> LLMResponse:
        gen_cfg: dict = {
            "temperature": temp,
            "max_output_tokens": mtok,
        }
        if self.thinking_level:
            gen_cfg["thinking_level"] = self.thinking_level
        body_obj: dict = {
            "store": False,
            "generation_config": gen_cfg,
        }
        cmd = [
            "gemini-api",
            "generate",
            "-m",
            self.model,
            "--stdin",
            "--stream=false",
            "-o",
            "json",
            "--no-interactive",
            "--body",
            json.dumps(body_obj),
        ]
        if system:
            cmd += ["--system", system]
        last_err = ""
        for i in range(retries):
            try:
                p = subprocess.run(
                    cmd,
                    input=user,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                last_err = f"gemini-api timeout ({self.timeout}s)"
                if i < retries - 1:
                    time.sleep(min(2 ** i, 16))
                    continue
                return LLMResponse(text="", model=self.model, raw_error=last_err)
            if p.returncode != 0:
                last_err = (p.stderr or p.stdout).strip()[:400]
                if i < retries - 1:
                    time.sleep(min(2 ** i, 16))
                    continue
                return LLMResponse(text="", model=self.model, raw_error=last_err)
            try:
                data = json.loads(p.stdout)
            except json.JSONDecodeError as e:
                last_err = f"bad JSON from gemini-api: {e} | {p.stdout[:200]}"
                if i < retries - 1:
                    time.sleep(min(2 ** i, 16))
                    continue
                return LLMResponse(text="", model=self.model, raw_error=last_err)
            u = data.get("usage") or {}
            text = _extract_interaction_text(data)
            return LLMResponse(
                text=text,
                input_tokens=int(u.get("total_input_tokens") or 0),
                output_tokens=int(u.get("total_output_tokens") or 0),
                thought_tokens=int(u.get("total_thought_tokens") or 0),
                model=self.model,
            )
        return LLMResponse(text="", model=self.model, raw_error=last_err)

    def _call_gemini_rest(self, system: str, user: str, temp: float, mtok: int, retries: int) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(text="", model=self.model, raw_error="GEMINI_API_KEY is not set")
        url = "https://generativelanguage.googleapis.com/v1/interactions"
        gen_cfg: dict = {
            "temperature": temp,
            "max_output_tokens": mtok,
        }
        if self.thinking_level:
            gen_cfg["thinking_level"] = self.thinking_level
        payload: dict = {
            "model": self.model,
            "input": user,
            "store": False,
            "generation_config": gen_cfg,
        }
        if system:
            payload["system_instruction"] = system
        body = json.dumps(payload).encode()
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        last_err = ""
        for i in range(retries):
            req = urllib.request.Request(url, body, headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                u = data.get("usage") or {}
                return LLMResponse(
                    text=_extract_interaction_text(data),
                    input_tokens=int(u.get("total_input_tokens") or 0),
                    output_tokens=int(u.get("total_output_tokens") or 0),
                    thought_tokens=int(u.get("total_thought_tokens") or 0),
                    model=self.model,
                )
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
                if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                    time.sleep(min(2 ** i, 16))
                    continue
                return LLMResponse(text="", model=self.model, raw_error=last_err)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = str(e)
                if i < retries - 1:
                    time.sleep(min(2 ** i, 16))
                    continue
                return LLMResponse(text="", model=self.model, raw_error=last_err)
        return LLMResponse(text="", model=self.model, raw_error=last_err)
