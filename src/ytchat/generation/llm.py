"""LLM adapters — all free-tier options behind one Protocol.

Three of the four hosted providers speak the OpenAI chat-completions dialect,
so ``OpenAICompatibleLLM`` covers Groq, OpenRouter and local Ollama with only a
base URL difference.  Gemini gets its own adapter because its SDK is distinct.

Free tiers rate-limit aggressively, so every adapter goes through a shared
spacing limiter plus exponential backoff on 429/503.  Without this, a Stage 4
eval run over 60 questions dies about eight questions in.
"""

from __future__ import annotations

import random
import re
import threading
import time
from typing import Protocol, Sequence, runtime_checkable

from ytchat.config import OPENAI_COMPATIBLE_BASE_URLS, Settings
from ytchat.errors import ConfigurationError, LLMError


@runtime_checkable
class LLM(Protocol):
    @property
    def model_id(self) -> str: ...

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> str: ...


class RateLimiter:
    """Minimum spacing between calls.  Thread-safe; a no-op when rpm <= 0."""

    def __init__(self, requests_per_minute: int) -> None:
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


_TRANSIENT = re.compile(
    r"429|rate.?limit|quota|resource.?exhausted|503|overloaded|timeout|unavailable",
    re.IGNORECASE,
)


# Conditions a retry can never clear inside a run: a daily token or request
# budget resets on a 24h boundary, and a free tier with no allowance for a model
# reports "limit: 0".  Both match _TRANSIENT on "429"/"rate limit", so without
# this check the client burns its full backoff schedule and fails anyway.
_EXHAUSTED = re.compile(
    r"tokens per day|requests per day|\bTPD\b|\bRPD\b|per day|limit:?\s*0\b",
    re.IGNORECASE,
)


def _is_transient(exc: Exception) -> bool:
    message = f"{type(exc).__name__} {exc}"
    if _EXHAUSTED.search(message):
        return False
    return bool(_TRANSIENT.search(message))


class _BaseLLM:
    def __init__(self, model: str, rpm: int, max_retries: int, timeout_s: float) -> None:
        self.model = model
        self.limiter = RateLimiter(rpm)
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    def _call_with_retries(self, fn) -> str:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                return fn()
            except Exception as exc:
                last = exc
                if not _is_transient(exc) or attempt == self.max_retries - 1:
                    break
                # Full jitter: free tiers punish synchronised retries.
                delay = min(2.0 ** attempt, 16.0) * (0.5 + random.random() / 2)
                time.sleep(delay)
        raise LLMError(f"{self.model_id} request failed: {last}") from last

    @property
    def model_id(self) -> str:
        return self.model


class GeminiLLM(_BaseLLM):
    """Google AI Studio free tier.  Supports both the new ``google-genai`` SDK
    and the legacy ``google-generativeai`` package."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", rpm: int = 12,
                 max_retries: int = 4, timeout_s: float = 60.0,
                 max_output_tokens: int = 1024) -> None:
        super().__init__(model, rpm, max_retries, timeout_s)
        if not api_key:
            raise ConfigurationError(
                "No Gemini API key. Get a free one at https://aistudio.google.com/apikey "
                "then set YTCHAT_GOOGLE_API_KEY (or put it in .env)."
            )
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self._client = None
        self._flavour = ""

    @property
    def model_id(self) -> str:
        return f"gemini:{self.model}"

    def _load(self):
        if self._client is not None:
            return
        try:
            from google import genai  # google-genai (current)

            self._client = genai.Client(api_key=self.api_key)
            self._flavour = "genai"
            return
        except ImportError:
            pass
        try:
            import google.generativeai as legacy  # google-generativeai (older)

            legacy.configure(api_key=self.api_key)
            self._client = legacy
            self._flavour = "legacy"
        except ImportError as exc:
            raise ConfigurationError(
                "Gemini support needs an SDK: pip install google-genai"
            ) from exc

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        self._load()

        def _run() -> str:
            if self._flavour == "genai":
                from google.genai import types

                resp = self._client.models.generate_content(  # type: ignore[union-attr]
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                return (resp.text or "").strip()

            model = self._client.GenerativeModel(  # type: ignore[union-attr]
                self.model, system_instruction=system
            )
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
            )
            return (resp.text or "").strip()

        return self._call_with_retries(_run)


class OpenAICompatibleLLM(_BaseLLM):
    """Groq, OpenRouter, and local Ollama — one dialect, three base URLs."""

    def __init__(self, provider: str, api_key: str, model: str, rpm: int = 15,
                 max_retries: int = 4, timeout_s: float = 60.0,
                 max_output_tokens: int = 1024, base_url: str | None = None) -> None:
        super().__init__(model, rpm, max_retries, timeout_s)
        self.provider = provider
        self.base_url = base_url or OPENAI_COMPATIBLE_BASE_URLS.get(provider)
        if not self.base_url:
            raise ConfigurationError(f"No base URL known for provider {provider!r}.")
        if not api_key:
            raise ConfigurationError(
                f"No API key for {provider}. Free keys: "
                "Groq → console.groq.com/keys, OpenRouter → openrouter.ai/keys. "
                f"Set YTCHAT_{provider.upper()}_API_KEY."
            )
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self._client = None

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"

    def _load(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigurationError(
                    f"{self.provider} support needs the openai SDK: pip install openai"
                ) from exc
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s
            )
        return self._client

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        client = self._load()

        def _run() -> str:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=self.max_output_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        return self._call_with_retries(_run)


class ScriptedLLM:
    """Offline test double: returns queued responses, records every prompt.

    Falls back to a refusal when the queue empties — so a test that triggers an
    unexpected extra call fails loudly instead of hanging on a network call.
    """

    def __init__(self, responses: Sequence[str] | None = None, default: str | None = None) -> None:
        self.responses = list(responses or [])
        self.default = default if default is not None else "INSUFFICIENT_CONTEXT"
        self.calls: list[dict[str, str]] = []

    @property
    def model_id(self) -> str:
        return "scripted"

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "prompt": prompt})
        return self.responses.pop(0) if self.responses else self.default


def build_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider
    if provider == "scripted":
        return ScriptedLLM()
    common = dict(
        model=settings.resolved_model,
        rpm=settings.resolved_rpm,
        max_retries=settings.llm_max_retries,
        timeout_s=settings.llm_timeout_s,
        max_output_tokens=settings.max_output_tokens,
    )
    if provider == "gemini":
        return GeminiLLM(api_key=settings.api_key_for("gemini") or "", **common)
    if provider in OPENAI_COMPATIBLE_BASE_URLS:
        return OpenAICompatibleLLM(
            provider=provider, api_key=settings.api_key_for(provider) or "", **common
        )
    raise ConfigurationError(f"Unknown LLM provider: {provider!r}")