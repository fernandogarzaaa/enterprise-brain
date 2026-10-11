"""Chat-completion and embedding providers.

Chat:
  * ``none``   - no LLM; answers are extractive (default when no LLM is configured).
  * ``mock``   - deterministic offline "LLM" for tests, demos and CI.
  * ``openai`` - any OpenAI-compatible ``/chat/completions`` API: OpenAI, Groq,
                 OpenRouter, Together, a local Ollama or vLLM server, ...

Embeddings:
  * ``hashed`` - deterministic feature-hashed bag of words (numpy only, offline).
  * ``openai`` - any OpenAI-compatible ``/embeddings`` API.
  * ``st``     - sentence-transformers, if installed (optional extra).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
import numpy as np

from enterprise_brain.config import Settings

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """The language-model provider failed or returned an unusable response."""


class ChatProvider(Protocol):
    name: str

    def complete(self, messages: Sequence[Message]) -> str: ...


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CITE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


# ---------------------------------------------------------------- chat providers


class MockChatProvider:
    """Deterministic stand-in for an LLM.

    It reads the numbered source blocks of the prompt built by ``ask`` and answers with
    the sentence that best matches the question, citing its source. Same input, same output.
    """

    name = "mock"

    def complete(self, messages: Sequence[Message]) -> str:
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if not user.strip():
            raise LLMError("empty prompt")
        question = user.rsplit("Question:", 1)[-1]
        q_terms = set(tokens(question))
        best: tuple[int, int, str] | None = None
        for line in user.splitlines():
            m = _CITE_RE.match(line.strip())
            if not m:
                continue
            n, body = int(m.group(1)), m.group(2)
            _, sep, rest = body.partition(": ")  # drop the "Source name: " label
            body = rest if sep else body
            for sent in sentences(body):
                score = len(q_terms & set(tokens(sent)))
                if best is None or score > best[0]:
                    best = (score, n, sent)
        if best is None or best[0] == 0:
            return "The provided sources do not answer this question."
        _, n, sent = best
        sent = sent if sent.endswith((".", "!", "?")) else sent + "."
        return f"{sent} [{n}]"


_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def post_json_with_retries(
    client: httpx.Client,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    max_retries: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    """POST JSON, retrying transient failures with exponential backoff."""
    last_error = "no attempt made"
    for attempt in range(max_retries + 1):
        if attempt:
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
        try:
            resp = client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning("LLM request failed", extra={"attempt": attempt + 1, "error": last_error})
            continue
        if resp.status_code in _RETRYABLE_STATUS:
            last_error = f"HTTP {resp.status_code}"
            log.warning("LLM transient error", extra={"attempt": attempt + 1, "status": resp.status_code})
            continue
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError("response is not JSON") from exc
        if not isinstance(data, dict):
            raise LLMError("response is not a JSON object")
        return data
    raise LLMError(f"giving up after {max_retries + 1} attempts: {last_error}")


def _auth_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:  # local servers such as Ollama need no key
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class OpenAICompatibleChatProvider:
    """Chat completions over HTTP with timeouts and bounded retries."""

    name = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 800,
        client: httpx.Client | None = None,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backoff_seconds = backoff_seconds
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, messages: Sequence[Message]) -> str:
        body = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = post_json_with_retries(
            self._client,
            f"{self.base_url}/chat/completions",
            body,
            _auth_headers(self.api_key),
            self.max_retries,
            self.backoff_seconds,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed chat response: {exc!r}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("empty chat response")
        return content.strip()


def get_chat_provider(settings: Settings) -> ChatProvider | None:
    mode = settings.llm_mode
    if mode == "none":
        return None
    if mode == "mock":
        return MockChatProvider()
    return OpenAICompatibleChatProvider(
        settings.LLM_BASE_URL,
        settings.LLM_API_KEY.get_secret_value(),
        settings.LLM_CHAT_MODEL,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


# ---------------------------------------------------------------- embedders


class HashedEmbedder:
    """Feature-hashed, L2-normalised bag of words. Deterministic and offline.

    Tokens are bucketed by MD5 (a non-security use) so vectors match those of 0.x
    beta databases, which therefore need no re-indexing after upgrading.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_id = f"hashed-{dim}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float64)
            for tok in tokens(text):
                h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
                v[h % self.dim] += 1.0
            norm = float(np.linalg.norm(v))
            out.append((v / norm).tolist() if norm > 0 else v.tolist())
        return out


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        batch_size: int = 64,
        client: httpx.Client | None = None,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.model_id = f"openai:{model}"
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.backoff_seconds = backoff_seconds
        self._client = client or httpx.Client(timeout=timeout)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            data = post_json_with_retries(
                self._client,
                f"{self.base_url}/embeddings",
                {"model": self.model, "input": batch},
                _auth_headers(self.api_key),
                self.max_retries,
                self.backoff_seconds,
            )
            try:
                rows = sorted(data["data"], key=lambda d: d.get("index", 0))
                vectors = [[float(x) for x in row["embedding"]] for row in rows]
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise LLMError(f"malformed embeddings response: {exc!r}") from exc
            if len(vectors) != len(batch):
                raise LLMError(f"expected {len(batch)} embeddings, got {len(vectors)}")
            out.extend(vectors)
        return out


class SentenceTransformerEmbedder:  # pragma: no cover - optional heavy dependency
    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LLMError("EMBEDDINGS=st requires `pip install enterprise-brain[st]`") from exc
        self._model = SentenceTransformer(model)
        self.model_id = f"st:{model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]


_st_cache: dict[str, Embedder] = {}


def get_embedder(settings: Settings) -> Embedder:
    if settings.EMBEDDINGS == "hashed":
        return HashedEmbedder(settings.HASHED_EMBED_DIM)
    if settings.EMBEDDINGS == "openai":
        return OpenAICompatibleEmbedder(
            settings.embed_base_url,
            settings.embed_api_key,
            settings.EMBED_MODEL,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    if settings.ST_MODEL not in _st_cache:  # pragma: no cover - optional dependency
        _st_cache[settings.ST_MODEL] = SentenceTransformerEmbedder(settings.ST_MODEL)
    return _st_cache[settings.ST_MODEL]  # pragma: no cover


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)
