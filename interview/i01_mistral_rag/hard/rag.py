"""RAG over Wikipedia + Mistral, with raw HTTP (httpx). HARD MODE stub.

    cp interview/i01_mistral_rag/hard/rag.py interview/i01_mistral_rag/rag.py

Same public API as the guided stub, without the step-by-step docstrings. You decide the
request details, the error taxonomy, the retry policy and the system prompt. TASK.md ("Hard
mode") lists the minimum contract the tests rely on; everything else is your judgment.
Implement every `raise NotImplementedError`. Add helpers, constants and exception subclasses
freely, but keep these names and signatures (the tests import them).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_WIKIPEDIA_BASE_URL = "https://en.wikipedia.org"
DEFAULT_MODEL = "mistral-small-latest"
# Wikimedia's User-Agent policy wants a way to reach you: for real traffic, replace the URL below
# with your own contact URL or email (https://meta.wikimedia.org/wiki/User-Agent_policy).
DEFAULT_USER_AGENT = (
    "i01-mistral-rag/0.1 (interview practice; https://github.com/example/pytorch-training)"
)
NO_ANSWER = "I don't know: no relevant sources were found."
SYSTEM_PROMPT = ""  # Write your own. TASK.md "Hard mode" says what the tests check in it.

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


# --------------------------------------------------------------------------- errors


class RAGError(Exception):
    """Base class for every error this module raises on purpose."""


class ConfigurationError(RAGError):
    """Missing or invalid configuration."""


class ResponseFormatError(RAGError):
    """A 2xx response whose body breaks the API contract."""


class APIError(RAGError):
    """An HTTP request that failed."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class RetryableError(APIError):
    """A transient failure that was not recovered."""

    def __init__(
        self, message: str, *, status_code: int | None = None, body: str = "", attempts: int = 0
    ) -> None:
        super().__init__(message, status_code=status_code, body=body)
        self.attempts = attempts


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Document:
    title: str
    key: str  # Wikipedia page key, e.g. "Retrieval-augmented_generation"
    url: str
    text: str
    description: str = ""


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ChatResult:
    id: str
    model: str
    content: str
    finish_reason: str | None
    usage: Usage


@dataclass(frozen=True)
class Source:
    index: int  # the number the model cites, [index]
    title: str
    url: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]
    cited: list[int]
    usage: Usage | None
    finish_reason: str | None = None


class Retriever(Protocol):
    def search(self, query: str, limit: int = 5) -> list[Document]: ...


class ChatModel(Protocol):
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult: ...


# --------------------------------------------------------------------------- stage 1


def strip_html(text: str) -> str:
    """Turn a Wikipedia search excerpt into plain text."""
    raise NotImplementedError("Your code here")


class WikipediaClient:
    """Sync client for the Wikipedia REST search API."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        base_url: str = DEFAULT_WIKIPEDIA_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 10.0,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep
        self._rng = rng

    def search(self, query: str, limit: int = 5) -> list[Document]:
        """Full-text search: up to `limit` Documents in the API's rank order."""
        raise NotImplementedError("Your code here")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# --------------------------------------------------------------------------- stage 2


class MistralClient:
    """Sync client for Mistral chat completions."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_MISTRAL_BASE_URL,
        model: str = DEFAULT_MODEL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 30.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("MISTRAL_API_KEY")
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep
        self._rng = rng

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """One non-streaming chat completion."""
        raise NotImplementedError("Your code here")

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stage 4. Streaming chat completion: yields content deltas as they arrive."""
        raise NotImplementedError("Your code here")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# --------------------------------------------------------------------------- stage 3


def build_context(docs: Sequence[Document], max_chars: int) -> tuple[str, list[Source]]:
    """Pack documents into a numbered context under a character budget."""
    raise NotImplementedError("Your code here")


def build_messages(question: str, context: str) -> list[Message]:
    """The chat messages for one RAG question."""
    raise NotImplementedError("Your code here")


def extract_citations(text: str, n_sources: int) -> list[int]:
    """The valid source indices cited as [k] in `text`."""
    raise NotImplementedError("Your code here")


class RAGPipeline:
    """retrieve -> pack context -> prompt -> LLM -> Answer with citations."""

    def __init__(
        self,
        retriever: Retriever,
        llm: ChatModel,
        *,
        top_k: int = 5,
        max_context_chars: int = 4000,
        temperature: float | None = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, question: str) -> Answer:
        """Answer `question` from retrieved sources."""
        raise NotImplementedError("Your code here")


# --------------------------------------------------------------------------- stage 4: retries


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Seconds to wait from a Retry-After header value, or None."""
    raise NotImplementedError("Your code here")


def compute_backoff(
    attempt: int,
    *,
    base: float = 0.5,
    cap: float = 8.0,
    rng: Callable[[], float] = random.random,
) -> float:
    """Exponential backoff with full jitter."""
    raise NotImplementedError("Your code here")


def retry_delay(
    response: httpx.Response | None,
    attempt: int,
    *,
    base: float = 0.5,
    cap: float = 8.0,
    max_retry_after: float = 60.0,
    rng: Callable[[], float] = random.random,
) -> float:
    """How long to sleep before retry number `attempt`."""
    raise NotImplementedError("Your code here")


def send_with_retries(
    send: Callable[[], httpx.Response],
    *,
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    backoff_base: float = 0.5,
    backoff_cap: float = 8.0,
    max_retry_after: float = 60.0,
) -> httpx.Response:
    """Call `send()` (one HTTP request), retrying transient failures."""
    raise NotImplementedError("Your code here")


async def async_send_with_retries(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    max_retries: int,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
    backoff_base: float = 0.5,
    backoff_cap: float = 8.0,
    max_retry_after: float = 60.0,
) -> httpx.Response:
    """Async send_with_retries."""
    raise NotImplementedError("Your code here")


# --------------------------------------------------------------------------- stage 4: async


class AsyncWikipediaClient:
    """Async client used to fetch full page summaries concurrently."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        base_url: str = DEFAULT_WIKIPEDIA_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 10.0,
        max_retries: int = 2,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep
        self._rng = rng

    async def summary(self, key: str) -> Document:
        """The page summary for one Wikipedia page key."""
        raise NotImplementedError("Your code here")

    async def enrich(self, docs: Sequence[Document], *, max_concurrency: int = 4) -> list[Document]:
        """Replace each doc by its summary, fetched concurrently."""
        raise NotImplementedError("Your code here")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


# --------------------------------------------------------------------------- stage 4: streaming


def parse_sse(lines: Iterable[str]) -> Iterator[str]:
    """Yield content deltas from Mistral's Server-Sent-Event lines."""
    raise NotImplementedError("Your code here")
