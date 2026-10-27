"""Offline tests for rag.py. No network, no API key: every HTTP call hits an httpx.MockTransport.

Each handler asserts on the request your code sends (method, URL, params, headers, JSON body)
and returns a canned response, so these tests check the HTTP contract, not just return values.
Run one stage at a time: `uv run pytest interview/i01_mistral_rag -rA -k stage1`.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest

from interview.i01_mistral_rag.rag import (
    DEFAULT_MODEL,
    NO_ANSWER,
    APIError,
    AsyncWikipediaClient,
    ChatResult,
    ConfigurationError,
    Document,
    MistralClient,
    RAGPipeline,
    ResponseFormatError,
    RetryableError,
    Source,
    Usage,
    WikipediaClient,
    async_send_with_retries,
    build_context,
    build_messages,
    compute_backoff,
    extract_citations,
    parse_retry_after,
    parse_sse,
    retry_delay,
    send_with_retries,
    strip_html,
)

WIKI = "https://en.wikipedia.org"
MISTRAL = "https://api.mistral.ai/v1"
UA = "test-agent/1.0 (tests@example.com)"
KEY = "sk-test-123"

Handler = Callable[[httpx.Request], httpx.Response]


def sync_client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class Recorder:
    """Collects sleep durations so backoff is asserted without actually waiting."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def wiki(handler: Handler, **kwargs: Any) -> WikipediaClient:
    kwargs.setdefault("sleep", Recorder())
    return WikipediaClient(sync_client(handler), user_agent=UA, **kwargs)


def mistral(handler: Handler, **kwargs: Any) -> MistralClient:
    kwargs.setdefault("sleep", Recorder())
    kwargs.setdefault("api_key", KEY)
    return MistralClient(sync_client(handler), **kwargs)


SEARCH_JSON = {
    "pages": [
        {
            "id": 1,
            "key": "Retrieval-augmented_generation",
            "title": "Retrieval-augmented generation",
            "excerpt": '<span class="searchmatch">Retrieval</span>-augmented generation '
            "(RAG) &amp; friends &quot;ground&quot; LLMs",
            "matched_title": None,
            "description": "Technique for LLMs",
            "thumbnail": None,
        },
        {
            "id": 2,
            "key": "Large_language_model",
            "title": "Large language model",
            "excerpt": "A <b>large</b>\n  language   model",
            "matched_title": None,
            "description": None,
            "thumbnail": None,
        },
    ]
}


def chat_json(content: str = "RAG grounds answers [1].") -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "mistral-small-latest",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


def ok_chat(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=chat_json())


USER_MSG = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- stage 1: Wikipedia


def test_stage1_strip_html_removes_tags_unescapes_and_collapses_whitespace() -> None:
    raw = (
        '<span class="searchmatch">Paris</span> is the  capital &amp; &quot;city&quot;\n of France'
    )
    assert strip_html(raw) == 'Paris is the capital & "city" of France'


def test_stage1_strip_html_keeps_escaped_angle_brackets_as_text() -> None:
    # Unescape AFTER stripping tags, or the literal "<b>" written by the author disappears.
    assert strip_html("use &lt;b&gt; for bold") == "use <b> for bold"
    assert strip_html("") == ""


def test_stage1_search_sends_expected_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=SEARCH_JSON)

    wiki(handler, timeout=3.0).search("  what is RAG ", limit=2)
    (request,) = seen
    assert request.method == "GET"
    assert request.url.host == "en.wikipedia.org"
    assert request.url.path == "/w/rest.php/v1/search/page"
    assert request.url.params["q"] == "what is RAG"
    assert request.url.params["limit"] == "2"
    assert request.headers["User-Agent"] == UA
    assert request.headers["Accept"] == "application/json"
    assert request.extensions["timeout"]["read"] == 3.0


def test_stage1_search_parses_documents_in_rank_order() -> None:
    docs = wiki(lambda r: httpx.Response(200, json=SEARCH_JSON)).search("rag", limit=2)
    assert docs == [
        Document(
            title="Retrieval-augmented generation",
            key="Retrieval-augmented_generation",
            url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            text='Retrieval-augmented generation (RAG) & friends "ground" LLMs',
            description="Technique for LLMs",
        ),
        Document(
            title="Large language model",
            key="Large_language_model",
            url="https://en.wikipedia.org/wiki/Large_language_model",
            text="A large language model",
            description="",
        ),
    ]


def test_stage1_search_uses_base_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("http://wiki.local/w/rest.php/v1/search/page?")
        return httpx.Response(200, json={"pages": [SEARCH_JSON["pages"][0]]})

    (doc,) = wiki(handler, base_url="http://wiki.local/").search("rag")
    assert doc.url == "http://wiki.local/wiki/Retrieval-augmented_generation"


def test_stage1_search_empty_results() -> None:
    assert wiki(lambda r: httpx.Response(200, json={"pages": []})).search("zzqx") == []


def test_stage1_search_validates_arguments_without_sending() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent for invalid arguments")

    client = wiki(handler)
    for bad_query in ["", "   "]:
        with pytest.raises(ValueError):
            client.search(bad_query)
    for bad_limit in [0, 101]:
        with pytest.raises(ValueError):
            client.search("rag", limit=bad_limit)


def test_stage1_search_non_200_raises_api_error_with_status_and_body() -> None:
    client = wiki(lambda r: httpx.Response(404, text="no such endpoint"))
    with pytest.raises(APIError) as info:
        client.search("rag")
    assert info.value.status_code == 404
    assert "no such endpoint" in info.value.body
    assert "404" in str(info.value)


def test_stage1_search_malformed_json_raises_response_format_error() -> None:
    client = wiki(lambda r: httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(ResponseFormatError):
        client.search("rag")


@pytest.mark.parametrize(
    "payload",
    [{}, {"pages": "nope"}, {"pages": [{"title": "no key"}]}, {"pages": [{"key": "K"}]}, []],
    ids=["no-pages", "pages-not-list", "page-without-key", "page-without-title", "not-a-dict"],
)
def test_stage1_search_missing_fields_raise_response_format_error(payload: Any) -> None:
    client = wiki(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(ResponseFormatError):
        client.search("rag")


# --------------------------------------------------------------------------- stage 2: Mistral


def test_stage2_chat_sends_auth_and_minimal_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=chat_json())

    mistral(handler, user_agent=UA, timeout=7.0).chat(USER_MSG)
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == f"{MISTRAL}/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Accept"] == "application/json"
    assert request.headers["User-Agent"] == UA
    assert request.extensions["timeout"]["read"] == 7.0
    # Unset options are omitted, not sent as null.
    assert json.loads(request.content) == {"model": DEFAULT_MODEL, "messages": USER_MSG}


def test_stage2_chat_sends_options_and_custom_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "model": "mistral-large-latest",
            "messages": USER_MSG,
            "temperature": 0.0,
            "max_tokens": 64,
        }
        return httpx.Response(200, json=chat_json())

    mistral(handler, model="mistral-large-latest").chat(USER_MSG, temperature=0.0, max_tokens=64)


def test_stage2_chat_parses_result_and_usage() -> None:
    result = mistral(ok_chat).chat(USER_MSG)
    assert result == ChatResult(
        id="cmpl-1",
        model="mistral-small-latest",
        content="RAG grounds answers [1].",
        finish_reason="stop",
        usage=Usage(prompt_tokens=12, completion_tokens=5, total_tokens=17),
    )


def test_stage2_api_key_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-from-env")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-from-env"
        return httpx.Response(200, json=chat_json())

    MistralClient(sync_client(handler)).chat(USER_MSG)


@pytest.mark.parametrize("env_value", [None, ""], ids=["unset", "empty"])
def test_stage2_missing_api_key_raises_before_any_request(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    if env_value is None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    else:
        monkeypatch.setenv("MISTRAL_API_KEY", env_value)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API without a key")

    client = MistralClient(sync_client(handler))  # constructing without a key is fine
    with pytest.raises(ConfigurationError, match="MISTRAL_API_KEY"):
        client.chat(USER_MSG)


def test_stage2_chat_empty_messages_raises_value_error_without_sending() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent for empty messages")

    with pytest.raises(ValueError):
        mistral(handler).chat([])


def test_stage2_chat_error_status_raises_api_error_with_body() -> None:
    body = {"object": "error", "message": "Invalid model: nope", "type": "invalid_model"}
    client = mistral(lambda r: httpx.Response(400, json=body))
    with pytest.raises(APIError) as info:
        client.chat(USER_MSG)
    assert info.value.status_code == 400
    assert "Invalid model" in info.value.body


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("choices"),
        lambda d: d.update(choices=[]),
        lambda d: d["choices"][0].pop("message"),
        lambda d: d["choices"][0]["message"].update(content=None),
        lambda d: d.pop("usage"),
        lambda d: d["usage"].update(total_tokens="17"),
    ],
    ids=["no-choices", "empty-choices", "no-message", "null-content", "no-usage", "bad-usage"],
)
def test_stage2_chat_malformed_response_raises_response_format_error(
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    payload = chat_json()
    mutate(payload)
    client = mistral(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(ResponseFormatError):
        client.chat(USER_MSG)


def test_stage2_chat_non_json_body_raises_response_format_error() -> None:
    client = mistral(lambda r: httpx.Response(200, text="upstream proxy says hi"))
    with pytest.raises(ResponseFormatError):
        client.chat(USER_MSG)


# --------------------------------------------------------------------------- stage 3: RAG


def doc(n: int, text: str) -> Document:
    return Document(title=f"T{n}", key=f"K{n}", url=f"https://w/K{n}", text=text)


def test_stage3_build_context_numbers_documents_and_maps_sources() -> None:
    context, sources = build_context([doc(1, "alpha"), doc(2, "beta")], max_chars=1000)
    assert context == "[1] T1\nalpha\n\n[2] T2\nbeta"
    assert sources == [Source(1, "T1", "https://w/K1"), Source(2, "T2", "https://w/K2")]


def test_stage3_build_context_budget_keeps_whole_documents_in_rank_order() -> None:
    docs = [doc(1, "a" * 40), doc(2, "b" * 40), doc(3, "c" * 40), doc(4, "d")]
    full_two = "[1] T1\n" + "a" * 40 + "\n\n[2] T2\n" + "b" * 40  # 96 chars
    assert len(full_two) == 96
    for budget in [96, 97, 143]:  # doc 3 needs 2 + 47 = 49 more, so it fits from 145 on
        context, sources = build_context(docs, max_chars=budget)
        assert context == full_two, budget  # never a partial document
        assert [s.index for s in sources] == [1, 2]  # stops at the first misfit (no doc 4)
    context, sources = build_context(docs, max_chars=145)
    assert [s.index for s in sources] == [1, 2, 3]
    assert len(context) == 145


def test_stage3_build_context_nothing_fits() -> None:
    assert build_context([doc(1, "x" * 50)], max_chars=10) == ("", [])
    assert build_context([], max_chars=100) == ("", [])


def test_stage3_build_messages_system_prompt_and_user_turn() -> None:
    messages = build_messages("Who?", "[1] T1\nalpha")
    assert [m["role"] for m in messages] == ["system", "user"]
    system = messages[0]["content"].lower()
    for phrase in ["only", "sources", "cite", "[1]", "i don't know"]:
        assert phrase in system, phrase
    assert "[1] T1\nalpha" in messages[1]["content"]
    assert "Who?" in messages[1]["content"]
    assert "alpha" not in messages[0]["content"]  # retrieved text never goes in the system turn


def test_stage3_extract_citations_sorted_unique_and_in_range() -> None:
    assert extract_citations("A [2]. B [1][2]. C [7]. D [0]. E [x].", n_sources=3) == [1, 2]
    assert extract_citations("no citations", n_sources=3) == []


class FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 5) -> list[Document]:
        self.calls.append((query, limit))
        return self.docs[:limit]


class FakeLLM:
    def __init__(self, content: str, finish_reason: str | None = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.calls.append(
            {"messages": list(messages), "temperature": temperature, "max_tokens": max_tokens}
        )
        return ChatResult("id", "m", self.content, self.finish_reason, Usage(1, 2, 3))


def test_stage3_pipeline_answers_with_citations() -> None:
    retriever = FakeRetriever([doc(1, "alpha"), doc(2, "beta"), doc(3, "gamma")])
    llm = FakeLLM("Alpha is first [1] and beta second [2]. Also [9].")
    pipeline = RAGPipeline(retriever, llm, top_k=3, max_context_chars=1000, temperature=0.1)
    answer = pipeline.answer("What comes first?")

    assert retriever.calls == [("What comes first?", 3)]
    (call,) = llm.calls
    assert call["messages"] == build_messages(
        "What comes first?", build_context(retriever.docs, 1000)[0]
    )
    assert call["temperature"] == 0.1
    assert answer.text == "Alpha is first [1] and beta second [2]. Also [9]."
    assert [s.index for s in answer.sources] == [1, 2, 3]
    assert answer.cited == [1, 2]
    assert answer.usage == Usage(1, 2, 3)


@pytest.mark.parametrize("reason", ["length", "model_length"])
def test_stage3_pipeline_warns_and_exposes_truncated_answer(
    reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLM("Alpha is [1", finish_reason=reason)
    with caplog.at_level(logging.WARNING):
        answer = RAGPipeline(FakeRetriever([doc(1, "alpha")]), llm).answer("q")
    assert answer.finish_reason == reason
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_stage3_pipeline_complete_answer_has_stop_and_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        answer = RAGPipeline(FakeRetriever([doc(1, "alpha")]), FakeLLM("A [1].")).answer("q")
    assert answer.finish_reason == "stop"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_stage3_pipeline_respects_context_budget() -> None:
    retriever = FakeRetriever([doc(1, "a" * 40), doc(2, "b" * 40), doc(3, "c" * 40)])
    llm = FakeLLM("[1]")
    answer = RAGPipeline(retriever, llm, max_context_chars=100).answer("q")
    assert [s.index for s in answer.sources] == [1, 2]
    user_turn = llm.calls[0]["messages"][1]["content"]
    assert "b" * 40 in user_turn
    assert "c" * 40 not in user_turn


def test_stage3_pipeline_without_sources_says_i_dont_know_and_skips_llm() -> None:
    llm = FakeLLM("should never be called")
    answer = RAGPipeline(FakeRetriever([]), llm).answer("anything?")
    assert answer.text == NO_ANSWER
    assert answer.sources == [] and answer.cited == [] and answer.usage is None
    assert llm.calls == []


def test_stage3_pipeline_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        RAGPipeline(FakeRetriever([]), FakeLLM("")).answer("  ")


def test_stage3_end_to_end_with_real_clients_over_mock_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "en.wikipedia.org":
            return httpx.Response(200, json=SEARCH_JSON)
        assert request.url.host == "api.mistral.ai"
        user_turn = json.loads(request.content)["messages"][1]["content"]
        assert "[1] Retrieval-augmented generation\nRetrieval-augmented generation (RAG)" in (
            user_turn
        )
        return httpx.Response(200, json=chat_json("It grounds LLMs [1]."))

    http = sync_client(handler)
    pipeline = RAGPipeline(
        WikipediaClient(http, user_agent=UA), MistralClient(http, api_key=KEY), top_k=2
    )
    answer = pipeline.answer("What is RAG?")
    assert answer.cited == [1]
    assert answer.sources[0].url == "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"


# --------------------------------------------------------------------------- stage 4: resilience


def test_stage4_parse_retry_after() -> None:
    assert parse_retry_after("2") == 2.0
    assert parse_retry_after(" 1.5 ") == 1.5
    assert parse_retry_after("-3") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    date = "Wed, 21 Oct 2015 07:28:00 GMT"  # epoch 1445412480
    assert parse_retry_after(date, now=1445412480 - 4.0) == pytest.approx(4.0)
    assert parse_retry_after(date, now=1445412480 + 100.0) == 0.0
    naive = "Wed, 21 Oct 2015 07:28:00 -0000"  # no timezone: still UTC, not local time
    assert parse_retry_after(naive, now=1445412480 - 4.0) == pytest.approx(4.0)


def test_stage4_compute_backoff_is_exponential_capped_and_jittered() -> None:
    def one() -> float:
        return 1.0

    assert [compute_backoff(a, base=0.5, cap=3.0, rng=one) for a in range(4)] == [
        0.5,
        1.0,
        2.0,
        3.0,
    ]
    assert compute_backoff(2, base=0.5, cap=8.0, rng=lambda: 0.25) == 0.5  # full jitter
    assert compute_backoff(5000, base=0.5, cap=8.0, rng=one) == 8.0  # no OverflowError


def test_stage4_retry_delay_prefers_retry_after_and_caps_it() -> None:
    def resp(headers: dict[str, str]) -> httpx.Response:
        return httpx.Response(429, headers=headers)

    def one() -> float:
        return 1.0

    assert retry_delay(resp({"Retry-After": "3"}), 0, rng=one) == 3.0
    assert retry_delay(resp({"Retry-After": "3600"}), 0, max_retry_after=60.0, rng=one) == 60.0
    assert retry_delay(resp({}), 2, base=0.5, rng=one) == 2.0
    assert retry_delay(None, 1, base=0.5, rng=one) == 1.0


class Script:
    """A send() callable that replays a scripted list of responses / exceptions."""

    def __init__(self, *outcomes: httpx.Response | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self) -> httpx.Response:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        outcome.request = httpx.Request("GET", "https://example.test/x")
        return outcome


def one() -> float:
    return 1.0


Run = Callable[..., httpx.Response]


def run_sync(send: Script, sleep: Recorder, **kwargs: Any) -> httpx.Response:
    return send_with_retries(send, sleep=sleep, **kwargs)


def run_async(send: Script, sleep: Recorder, **kwargs: Any) -> httpx.Response:
    async def asend() -> httpx.Response:
        return send()

    async def asleep(seconds: float) -> None:
        sleep(seconds)

    return asyncio.run(async_send_with_retries(asend, sleep=asleep, **kwargs))


# Every retry-policy test runs against both send_with_retries and async_send_with_retries.
both = pytest.mark.parametrize("run", [run_sync, run_async], ids=["sync", "async"])


@both
def test_stage4_429_retry_after_is_honored(run: Run) -> None:
    sleep = Recorder()
    send = Script(httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200))
    response = run(send, sleep, max_retries=3, rng=one)
    assert response.status_code == 200
    assert send.calls == 2
    assert sleep.sleeps == [2.0]


@both
def test_stage4_retry_after_longer_than_cap_gives_up_without_sleeping(run: Run) -> None:
    sleep = Recorder()
    send = Script(httpx.Response(429, headers={"Retry-After": "3600"}, text="later"))
    with pytest.raises(RetryableError) as info:
        run(send, sleep, max_retries=3, rng=one, max_retry_after=60.0)
    assert send.calls == 1
    assert sleep.sleeps == []  # neither the full hour nor the 60 s cap
    assert info.value.status_code == 429
    assert info.value.attempts == 1


@both
def test_stage4_503_retried_with_exponential_backoff_then_success(run: Run) -> None:
    sleep = Recorder()
    send = Script(httpx.Response(503), httpx.Response(502), httpx.Response(200, text="ok"))
    response = run(send, sleep, max_retries=3, rng=one, backoff_base=0.5)
    assert response.text == "ok"
    assert sleep.sleeps == [0.5, 1.0]


@both
def test_stage4_backoff_uses_injected_rng(run: Run) -> None:
    sleep = Recorder()
    send = Script(httpx.Response(500), httpx.Response(504), httpx.Response(200))
    run(send, sleep, max_retries=2, rng=lambda: 0.5, backoff_base=1.0)
    assert sleep.sleeps == [0.5, 1.0]


@both
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_stage4_client_errors_are_not_retried(run: Run, status: int) -> None:
    sleep = Recorder()
    send = Script(httpx.Response(status, text="bad"), httpx.Response(200))
    with pytest.raises(APIError) as info:
        run(send, sleep, max_retries=3, rng=one)
    assert not isinstance(info.value, RetryableError)
    assert info.value.status_code == status
    assert send.calls == 1
    assert sleep.sleeps == []


@both
def test_stage4_max_retries_exhausted_raises_retryable_error(run: Run) -> None:
    sleep = Recorder()
    send = Script(*(httpx.Response(503, text="overloaded") for _ in range(3)))
    with pytest.raises(RetryableError) as info:
        run(send, sleep, max_retries=2, rng=one)
    assert send.calls == 3  # 1 try + 2 retries
    assert len(sleep.sleeps) == 2  # no pointless sleep after the last attempt
    assert info.value.status_code == 503
    assert info.value.attempts == 3
    assert "overloaded" in info.value.body


@both
def test_stage4_timeout_is_retried_then_succeeds(run: Run) -> None:
    sleep = Recorder()
    send = Script(httpx.ReadTimeout("slow"), httpx.ConnectError("refused"), httpx.Response(200))
    assert run(send, sleep, max_retries=2, rng=one).status_code == 200
    assert sleep.sleeps == [0.5, 1.0]


@both
def test_stage4_stale_keepalive_disconnect_is_retried(run: Run) -> None:
    sleep = Recorder()
    dropped = httpx.RemoteProtocolError("Server disconnected without sending a response.")
    send = Script(dropped, httpx.Response(200))
    assert run(send, sleep, max_retries=1, rng=one).status_code == 200
    assert send.calls == 2
    assert sleep.sleeps == [0.5]


@both
def test_stage4_timeout_exhausted_raises_retryable_error_with_cause(run: Run) -> None:
    sleep = Recorder()
    send = Script(httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"))
    with pytest.raises(RetryableError) as info:
        run(send, sleep, max_retries=1, rng=one)
    assert info.value.status_code is None
    assert info.value.attempts == 2
    assert isinstance(info.value.__cause__, httpx.TimeoutException)
    assert sleep.sleeps == [0.5]  # one retry, no sleep after the last attempt


def test_stage4_mistral_client_retries_429_honoring_retry_after() -> None:
    statuses = iter([429, 500, 200])
    sleep = Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {KEY}"  # on every attempt
        status = next(statuses)
        if status == 429:
            return httpx.Response(429, headers={"Retry-After": "1.5"}, json={"message": "slow"})
        if status == 500:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=chat_json())

    client = mistral(handler, sleep=sleep, rng=one, max_retries=3)
    assert client.chat(USER_MSG).content == "RAG grounds answers [1]."
    assert sleep.sleeps == [1.5, 1.0]  # Retry-After, then backoff attempt 1 (0.5 * 2)


def test_stage4_mistral_client_does_not_retry_401() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"message": "Unauthorized"})

    sleep = Recorder()
    with pytest.raises(APIError) as info:
        mistral(handler, sleep=sleep, max_retries=5).chat(USER_MSG)
    assert info.value.status_code == 401
    assert calls == 1 and sleep.sleeps == []


def test_stage4_mistral_timeout_retried_then_raises() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(RetryableError):
        mistral(handler, max_retries=2, rng=one).chat(USER_MSG)
    assert calls == 3


def test_stage4_wikipedia_client_retries_503() -> None:
    statuses = iter([503, 200])
    sleep = Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        if next(statuses) == 503:
            return httpx.Response(503)
        return httpx.Response(200, json=SEARCH_JSON)

    assert len(wiki(handler, sleep=sleep, rng=one).search("rag", limit=2)) == 2
    assert sleep.sleeps == [0.5]


def summary_json(key: str) -> dict[str, Any]:
    return {
        "title": key.replace("_", " "),
        "extract": f"Full summary of {key}.",
        "description": "desc",
        "content_urls": {"desktop": {"page": f"https://en.wikipedia.org/wiki/{key}"}},
    }


def test_stage4_async_summary_request_and_parse() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/rest_v1/page/summary/C++"
        assert request.url.raw_path == b"/api/rest_v1/page/summary/C%2B%2B"
        assert request.headers["User-Agent"] == UA
        return httpx.Response(200, json=summary_json("C++"))

    async def main() -> Document:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await AsyncWikipediaClient(http, user_agent=UA).summary("C++")

    assert asyncio.run(main()) == Document(
        title="C++",
        key="C++",
        url="https://en.wikipedia.org/wiki/C++",
        text="Full summary of C++.",
        description="desc",
    )


def run_summary(handler: Callable[..., Any], key: str) -> Document:
    async def main() -> Document:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await AsyncWikipediaClient(http, user_agent=UA).summary(key)

    return asyncio.run(main())


def test_stage4_async_summary_sends_timeout_and_follows_redirects() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.extensions["timeout"]["read"] == 10.0  # the client's default timeout
        if request.url.path.endswith("/Eiffel_tower"):  # a redirect title
            return httpx.Response(302, headers={"Location": "/api/rest_v1/page/summary/Eiffel"})
        return httpx.Response(200, json=summary_json("Eiffel"))

    assert run_summary(handler, "Eiffel_tower").text == "Full summary of Eiffel."


@pytest.mark.parametrize(
    "content_urls",
    [None, "weird", {"desktop": "weird"}, {"desktop": {"page": 7}}, {"mobile": {}}],
    ids=["missing", "not-a-dict", "desktop-not-a-dict", "page-not-a-str", "no-desktop"],
)
def test_stage4_async_summary_falls_back_to_key_url(content_urls: Any) -> None:
    payload: dict[str, Any] = {"title": "C++", "extract": "x"}
    if content_urls is not None:
        payload["content_urls"] = content_urls

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    assert run_summary(handler, "C++").url == "https://en.wikipedia.org/wiki/C%2B%2B"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>oops</html>"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"extract": "no title"}),
        httpx.Response(200, json={"title": "T", "extract": None}),
    ],
    ids=["not-json", "not-a-dict", "no-title", "null-extract"],
)
def test_stage4_async_summary_malformed_raises_response_format_error(
    response: httpx.Response,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return response

    with pytest.raises(ResponseFormatError):
        run_summary(handler, "K")


def test_stage4_async_enrich_rejects_bad_max_concurrency() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent")

    async def main() -> list[Document]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await AsyncWikipediaClient(http).enrich([doc(1, "x")], max_concurrency=0)

    with pytest.raises(ValueError):
        asyncio.run(main())


def test_stage4_async_enrich_bounds_concurrency_and_keeps_order() -> None:
    in_flight = 0
    max_in_flight = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        key = request.url.path.rsplit("/", 1)[-1]
        await asyncio.sleep(0.01 if key != "K0" else 0.03)  # first finishes last
        in_flight -= 1
        return httpx.Response(200, json=summary_json(key))

    docs = [doc(i, "snippet") for i in range(10)]
    docs = [Document(d.title, f"K{i}", d.url, d.text) for i, d in enumerate(docs)]

    async def main() -> list[Document]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await AsyncWikipediaClient(http, user_agent=UA).enrich(docs, max_concurrency=3)

    enriched = asyncio.run(main())
    assert max_in_flight == 3  # concurrent, but never more than the limit
    assert [d.key for d in enriched] == [f"K{i}" for i in range(10)]
    assert all(d.text == f"Full summary of K{i}." for i, d in enumerate(enriched))


def test_stage4_async_enrich_keeps_snippet_when_one_summary_fails_and_retries_async() -> None:
    attempts: dict[str, int] = {}
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path.rsplit("/", 1)[-1]
        attempts[key] = attempts.get(key, 0) + 1
        if key == "Missing":
            return httpx.Response(404, json={"title": "Not found."})
        if key == "Flaky" and attempts[key] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, json=summary_json(key))

    docs = [
        Document("Missing", "Missing", "u1", "snippet one"),
        Document("Flaky", "Flaky", "u2", "snippet two"),
    ]

    async def main() -> list[Document]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = AsyncWikipediaClient(http, user_agent=UA, sleep=fake_sleep, rng=one)
            return await client.enrich(docs)

    missing, flaky = asyncio.run(main())
    assert missing == docs[0]  # 404 is not retried; the search snippet is kept
    assert attempts["Missing"] == 1
    assert flaky.text == "Full summary of Flaky."
    assert sleeps == [0.25]


def test_stage4_parse_sse_yields_deltas_in_order_and_stops_at_done() -> None:
    def data(obj: Any) -> str:
        return "data: " + json.dumps(obj)

    lines = [
        ": keep-alive",
        data({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}),
        "",
        data({"choices": [{"index": 0, "delta": {"content": "Hel"}}]}),
        "",
        "event: message",
        data({"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]}),
        "",
        data({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
        "data: {this is not json and must never be parsed",
    ]
    assert list(parse_sse(lines)) == ["Hel", "lo"]


def test_stage4_parse_sse_malformed_chunk_raises() -> None:
    with pytest.raises(ResponseFormatError):
        list(parse_sse(["data: {not json"]))


def test_stage4_stream_end_to_end() -> None:
    chunks = ["The", " answer", " [1]"]
    sse = "".join(
        f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': c}}]})}\n\n"
        for c in chunks
    )
    sse += "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert request.headers["Accept"] == "text/event-stream"
        assert json.loads(request.content) == {
            "model": DEFAULT_MODEL,
            "messages": USER_MSG,
            "stream": True,
        }
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=sse.encode()
        )

    assert list(mistral(handler).stream(USER_MSG)) == chunks


def test_stage4_stream_sends_per_request_timeout() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    assert list(mistral(handler, timeout=7.0).stream(USER_MSG)) == []
    assert seen[0].extensions["timeout"]["read"] == 7.0


def test_stage4_stream_validates_eagerly_before_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API")

    # Raised by the call itself, not on the first next(): no list() around stream().
    with pytest.raises(ConfigurationError, match="MISTRAL_API_KEY"):
        MistralClient(sync_client(handler)).stream(USER_MSG)
    with pytest.raises(ValueError):
        mistral(handler).stream([])


def test_stage4_stream_error_status_raises_api_error() -> None:
    client = mistral(lambda r: httpx.Response(401, json={"message": "Unauthorized"}))
    with pytest.raises(APIError) as info:
        list(client.stream(USER_MSG))
    assert info.value.status_code == 401
    assert "Unauthorized" in info.value.body
