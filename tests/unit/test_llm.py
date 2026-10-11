from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import httpx
import pytest
import respx

from enterprise_brain import llm
from enterprise_brain.llm import (
    HashedEmbedder,
    LLMError,
    MockChatProvider,
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbedder,
    cosine,
    get_chat_provider,
    get_embedder,
    post_json_with_retries,
    sentences,
    tokens,
)

BASE = "https://llm.example.com/v1"


def test_tokens_and_sentences() -> None:
    assert tokens("Hello, World! ACME-42") == ["hello", "world", "acme", "42"]
    assert sentences("One. Two!  Three? Four") == ["One.", "Two!", "Three?", "Four"]
    assert sentences("   ") == []


# ---------------------------------------------------------------- embeddings


def test_hashed_embedder_is_normalised_and_deterministic() -> None:
    e = HashedEmbedder(64)
    assert e.model_id == "hashed-64"
    a, b, empty = e.embed(["acme contract", "acme contract", ""])
    assert a == b
    assert len(a) == 64
    assert math.isclose(sum(x * x for x in a), 1.0)
    assert empty == [0.0] * 64


def test_hashed_embedder_default_dim() -> None:
    assert HashedEmbedder().dim == 256
    assert HashedEmbedder().model_id == "hashed-256"


def test_hashed_embedder_counts_repeats() -> None:
    (v,) = HashedEmbedder(256).embed(["acme acme"])
    assert [x for x in v if x] == [1.0]
    (w,) = HashedEmbedder(256).embed(["invoice invoice approval"])
    assert sorted(x for x in w if x) == pytest.approx([1 / math.sqrt(5), 2 / math.sqrt(5)])


def test_hashed_embedder_matches_beta_vectors() -> None:
    # the 0.x beta bucketed tokens with md5 into 256 dims; 1.0 must match exactly
    (v,) = HashedEmbedder(256).embed(["Invoice"])
    idx = int(hashlib.md5(b"invoice").hexdigest(), 16) % 256
    assert v[idx] == 1.0


def test_hashed_similarity_ranks_related_text_higher() -> None:
    e = HashedEmbedder(256)
    q, near, far = e.embed(["acme contract status", "the acme contract is active", "leave policy for staff"])
    assert cosine(q, near) > cosine(q, far)


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0
    assert math.isclose(cosine([1, 1], [2, 2]), 1.0)
    assert math.isclose(cosine([1, 0], [-1, 0]), -1.0)
    assert math.isclose(cosine([3, 4], [4, 3]), 24 / 25)


@respx.mock
def test_openai_embedder_batches_and_orders() -> None:
    calls: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        assert request.headers["authorization"] == "Bearer k"
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})

    respx.post(f"{BASE}/embeddings").mock(side_effect=handler)
    e = OpenAICompatibleEmbedder(BASE + "/", "k", "m", batch_size=2, backoff_seconds=0)
    assert e.model_id == "openai:m"
    assert e.embed(["a", "bb", "ccc"]) == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert [c["input"] for c in calls] == [["a", "bb"], ["ccc"]]
    assert calls[0]["model"] == "m"


@respx.mock
def test_openai_embedder_defaults_and_no_key() -> None:
    route = respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"index": i, "embedding": [1]} for i in range(64)]})
    )
    e = OpenAICompatibleEmbedder(BASE, "", "m")
    assert (e.batch_size, e.max_retries) == (64, 2)
    assert len(e.embed(["x"] * 64)) == 64
    assert route.call_count == 1
    assert "authorization" not in route.calls.last.request.headers


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"nope": 1},
        {"data": [{"index": 0}]},
        {"data": [{"index": 0, "embedding": ["x"]}]},
        {"data": []},
        {"data": ["bad"]},
    ],
)
def test_openai_embedder_malformed(payload: dict[str, Any]) -> None:
    respx.post(f"{BASE}/embeddings").mock(return_value=httpx.Response(200, json=payload))
    with pytest.raises(LLMError):
        OpenAICompatibleEmbedder(BASE, "k", "m", max_retries=0).embed(["x"])


# ---------------------------------------------------------------- chat


def _chat_ok(content: str = "Answer [1]") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@respx.mock
def test_openai_chat_success_and_payload() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=_chat_ok("  Hi [1]  "))
    p = OpenAICompatibleChatProvider(BASE + "/", "key", "model-x", temperature=0.3, max_tokens=99)
    assert p.base_url == BASE
    assert p.complete([{"role": "user", "content": "q"}]) == "Hi [1]"
    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "model": "model-x",
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 0.3,
        "max_tokens": 99,
    }
    headers = route.calls.last.request.headers
    assert headers["authorization"] == "Bearer key"
    assert headers["content-type"] == "application/json"


def test_openai_chat_defaults() -> None:
    p = OpenAICompatibleChatProvider(BASE, "k", "m")
    assert (p.max_retries, p.temperature, p.max_tokens, p.backoff_seconds) == (2, 0.1, 800, 0.5)


@respx.mock
def test_openai_chat_retries_transient_then_succeeds() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[httpx.Response(429), httpx.ConnectError("boom"), _chat_ok()]
    )
    p = OpenAICompatibleChatProvider(BASE, "", "m", max_retries=2, backoff_seconds=0)
    assert p.complete([{"role": "user", "content": "q"}]) == "Answer [1]"
    assert route.call_count == 3
    assert "authorization" not in route.calls.last.request.headers


@respx.mock
@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
def test_openai_chat_gives_up_after_retries(status: int) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(status))
    p = OpenAICompatibleChatProvider(BASE, "k", "m", max_retries=1, backoff_seconds=0)
    with pytest.raises(LLMError, match=f"giving up after 2 attempts: HTTP {status}"):
        p.complete([{"role": "user", "content": "q"}])
    assert route.call_count == 2


@respx.mock
@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_openai_chat_client_error_not_retried(status: int) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(status, text="bad key"))
    p = OpenAICompatibleChatProvider(BASE, "k", "m", max_retries=3, backoff_seconds=0)
    with pytest.raises(LLMError, match=f"HTTP {status}: bad key"):
        p.complete([{"role": "user", "content": "q"}])
    assert route.call_count == 1


@respx.mock
def test_openai_chat_connect_errors_exhaust() -> None:
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    p = OpenAICompatibleChatProvider(BASE, "k", "m", max_retries=0, backoff_seconds=0)
    with pytest.raises(LLMError, match="1 attempts: ConnectError: down"):
        p.complete([{"role": "user", "content": "q"}])


@respx.mock
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, json={"choices": []}), "malformed"),
        (httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]}), "empty"),
        (httpx.Response(200, json={"choices": [{"message": {"content": None}}]}), "empty"),
        (httpx.Response(200, json=["not", "object"]), "not a JSON object"),
        (httpx.Response(200, text="<html>"), "not JSON"),
    ],
)
def test_openai_chat_malformed(response: httpx.Response, message: str) -> None:
    respx.post(f"{BASE}/chat/completions").mock(return_value=response)
    with pytest.raises(LLMError, match=message):
        OpenAICompatibleChatProvider(BASE, "k", "m", max_retries=0).complete([{"role": "user", "content": "q"}])


def test_post_json_backoff_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    with httpx.Client(transport=transport) as c, pytest.raises(LLMError, match="3 attempts"):
        post_json_with_retries(c, "https://x/y", {}, {}, max_retries=2, backoff_seconds=1.5)
    assert sleeps == [1.5, 3.0]


def test_post_json_zero_retries_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}))
    with httpx.Client(transport=transport) as c:
        assert post_json_with_retries(c, "https://x/y", {}, {}, 0, 5) == {"ok": True}
    assert sleeps == []


def test_post_json_success_on_399_boundary() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(399, json={"edge": 1}))
    with httpx.Client(transport=transport) as c:
        assert post_json_with_retries(c, "https://x/y", {}, {}, 0, 0) == {"edge": 1}


# ---------------------------------------------------------------- mock provider


def test_mock_chat_cites_best_source() -> None:
    prompt = (
        "Sources:\n[1] Leave policy: Employees get 20 days of leave.\n"
        "[2] Acme MSA: The Acme contract is active until 2027. It renews yearly.\n\n"
        "Question: Is the Acme contract active?"
    )
    out = MockChatProvider().complete([{"role": "system", "content": "s"}, {"role": "user", "content": prompt}])
    assert out == "The Acme contract is active until 2027. [2]"


def test_mock_chat_adds_period_and_handles_no_match() -> None:
    p = MockChatProvider()
    assert p.complete([{"role": "user", "content": "[1] Doc: acme rocks\n\nQuestion: acme"}]) == "acme rocks. [1]"
    unlabeled = p.complete([{"role": "user", "content": "[3] acme unlabeled\n\nQuestion: acme"}])
    assert unlabeled == "acme unlabeled. [3]"
    exclaim = p.complete([{"role": "user", "content": "[1] D: acme wins!\n\nQuestion: acme"}])
    assert exclaim == "acme wins! [1]"
    none = p.complete([{"role": "user", "content": "[1] Doc: hello\n\nQuestion: zebra"}])
    assert none == "The provided sources do not answer this question."
    nosrc = p.complete([{"role": "user", "content": "Question: zebra"}])
    assert nosrc == "The provided sources do not answer this question."


def test_mock_chat_uses_last_user_message() -> None:
    msgs = [
        {"role": "user", "content": "[1] A: zebra stripes.\n\nQuestion: zebra"},
        {"role": "user", "content": "[2] B: acme corp.\n\nQuestion: acme"},
    ]
    assert MockChatProvider().complete(msgs) == "acme corp. [2]"


def test_mock_chat_rejects_empty() -> None:
    with pytest.raises(LLMError):
        MockChatProvider().complete([{"role": "user", "content": "  "}])
    with pytest.raises(LLMError):
        MockChatProvider().complete([{"role": "system", "content": "only system"}])


def test_mock_chat_prefers_first_on_tie() -> None:
    prompt = "[1] A: acme one.\n[2] B: acme two.\n\nQuestion: acme"
    assert MockChatProvider().complete([{"role": "user", "content": prompt}]) == "acme one. [1]"


# ---------------------------------------------------------------- factories


def test_get_chat_provider_modes(settings_factory: Any) -> None:
    assert get_chat_provider(settings_factory(LLM_PROVIDER="none")) is None
    assert get_chat_provider(settings_factory(LLM_PROVIDER="auto")) is None
    assert isinstance(get_chat_provider(settings_factory(LLM_PROVIDER="mock")), MockChatProvider)
    p = get_chat_provider(
        settings_factory(
            LLM_PROVIDER="auto",
            LLM_BASE_URL=BASE,
            LLM_API_KEY="k",
            LLM_CHAT_MODEL="mx",
            LLM_MAX_RETRIES=4,
            LLM_TEMPERATURE=0.5,
            LLM_MAX_TOKENS=123,
        )
    )
    assert isinstance(p, OpenAICompatibleChatProvider)
    assert (p.base_url, p.api_key, p.model, p.max_retries, p.temperature, p.max_tokens) == (
        BASE,
        "k",
        "mx",
        4,
        0.5,
        123,
    )


def test_get_embedder(settings_factory: Any) -> None:
    e = get_embedder(settings_factory(HASHED_EMBED_DIM=128))
    assert isinstance(e, HashedEmbedder) and e.dim == 128
    o = get_embedder(
        settings_factory(EMBEDDINGS="openai", LLM_BASE_URL=BASE, LLM_API_KEY="k", EMBED_MODEL="emb", LLM_MAX_RETRIES=1)
    )
    assert isinstance(o, OpenAICompatibleEmbedder)
    assert (o.base_url, o.api_key, o.model, o.max_retries) == (BASE, "k", "emb", 1)
    o2 = get_embedder(
        settings_factory(EMBEDDINGS="openai", LLM_BASE_URL=BASE, EMBED_BASE_URL="http://emb:1", EMBED_API_KEY="ek")
    )
    assert isinstance(o2, OpenAICompatibleEmbedder)
    assert (o2.base_url, o2.api_key) == ("http://emb:1", "ek")
