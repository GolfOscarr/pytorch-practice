# i01: Wikipedia + Mistral RAG (pair programming, ~60 min)

Modeled on a real Mistral AI *Applied AI Engineer* pair-programming round (Oct 2025): "use a
third-party API for information retrieval in combination with the Mistral API, in order to add
the external information to the context." The candidate had no API key, couldn't test anything
end to end, ran out of time, and was rejected because their "skills with APIs were not meeting
expectations". This version runs fully offline, so you can get through all of it.

## What the interviewer says

> We'd like you to build a small question-answering service. A user asks a question; you search
> Wikipedia for relevant pages, put what you find into the prompt, and ask a Mistral model to
> answer **using only those sources, with citations** like `[1]`, `[2]`.
>
> Talk to raw HTTP with `httpx`, no SDKs. We care less about clever prompts than about how you
> handle a real API: auth, timeouts, error codes, retries, rate limits, validating what comes
> back, and keeping the code testable. Treat it like code you'd ship.
>
> There's no API key in this environment, so we've written offline tests that fake the HTTP
> layer. Work stage by stage, keep the tests green, and think out loud. Ask questions whenever
> you want.

Everything goes in **`rag.py`**. Names, signatures and dataclasses are already there, and the
docstrings say exactly what the tests check. Fill in each `raise NotImplementedError`.

That guided stub pre-decides every API judgment call, which is exactly what the real round
graded. Once you can pass it, or right away if you're preparing for an interview, use
**hard mode** (below).

## Stages

| Stage | Time | You build | Tests |
|---|---|---|---|
| 0 | ~5 min | your own `httpx.MockTransport` test for `search`, before reading `test_rag.py` | yours |
| 1 | ~15 min | `strip_html`, `WikipediaClient.search` → `list[Document]` | `-k stage1` |
| 2 | ~15 min | `MistralClient.chat` → `ChatResult` (with `Usage`) | `-k stage2` |
| 3 | ~15 min | `build_context`, `build_messages`, `extract_citations`, `RAGPipeline.answer` → `Answer` | `-k stage3` |
| 4 (stretch) | ~15 min | retries (`parse_retry_after`, `compute_backoff`, `retry_delay`, `send_with_retries`, async variant), `AsyncWikipediaClient.summary`/`enrich`, `parse_sse`, `MistralClient.stream` | `-k stage4` |

**Stage 0: test without a key, by yourself.** The real candidate failed partly because they
couldn't show anything worked without a key. Before you open `test_rag.py`, write one test of
your own for `WikipediaClient.search` in a new file (e.g. `interview/i01_mistral_rag/test_mine.py`):
a fake transport that asserts on the request you send and returns a canned body. Then compare
with the provided tests: what did they check that you didn't?

```python
import httpx

from interview.i01_mistral_rag.rag import WikipediaClient


def test_search_sends_query_and_parses_pages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)  # inspect method, URL, params, headers, timeout afterwards
        return httpx.Response(200, json={"pages": [{"key": "Paris", "title": "Paris"}]})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    docs = WikipediaClient(http).search("paris", limit=1)
    assert seen[0].url.params["q"] == "paris"
    assert docs[0].key == "Paris"
    # your turn: errors (404, 503, HTML with status 200), missing fields, empty query...
```

**Stage 1: retrieval.** `GET https://en.wikipedia.org/w/rest.php/v1/search/page?q=...&limit=...`.
Send a descriptive `User-Agent` (Wikimedia asks for one) and an explicit timeout. Validate
arguments before sending. Excerpts contain `<span class="searchmatch">` markup and HTML entities,
so strip and unescape them. Non-2xx → `APIError`; a body that isn't the documented shape →
`ResponseFormatError`.

**Stage 2: generation.** `POST https://api.mistral.ai/v1/chat/completions` with
`Authorization: Bearer $MISTRAL_API_KEY`. Default model `mistral-small-latest`. Missing key →
`ConfigurationError` before any request. Send only the options that were set. Parse `choices[0]`
and `usage` and validate them.

**Stage 3: RAG.** Number the sources, pack them into a context under a character budget
(**whole documents only, in rank order**), build a system prompt that says "only these sources,
cite them, say I don't know otherwise", call the LLM, and map the `[k]` citations back to
sources. If retrieval finds nothing, answer `NO_ANSWER` **without calling the LLM**.

**Stage 4: production hardening (stretch).**
- Retry 408/429/500/502/503/504, timeouts, connection errors and stale keep-alive disconnects
  (`httpx.RemoteProtocolError`) with exponential backoff and full jitter. Honor `Retry-After`
  (seconds or HTTP-date); if it asks for longer than `max_retry_after`, give up at once with
  `RetryableError` instead of sleeping. **Never retry other 4xx** (400, 401...). Plug it into
  both clients.
- Fetch full page summaries (`/api/rest_v1/page/summary/<key>`, following redirects)
  concurrently with `httpx.AsyncClient`, at most N in flight (`asyncio.Semaphore`), order
  preserved, and one failed page doesn't fail the batch.
- Streaming: parse `text/event-stream` (`data: {...}` lines, ending at `data: [DONE]`) and yield
  the deltas.

Also in stage 3: if the model stopped because it ran out of tokens (`finish_reason` is
`"length"` or `"model_length"`), log a warning and expose `finish_reason` on the `Answer`.

You don't have to finish stage 4. Talking through *how* you'd do it is worth a lot.

## Hard mode (recommended for interview prep)

```bash
cp interview/i01_mistral_rag/hard/rag.py interview/i01_mistral_rag/rag.py   # overwrites rag.py
```

Same names and signatures, one-line docstrings, and no system prompt. Work from this file and
TASK.md only: decide the request shape, the error taxonomy, which failures to retry and how, and
write the prompt yourself. Do stage 0 first. Don't read `test_rag.py` until a stage is done;
then run it and treat each failure as a review comment. The list below is the minimum contract
the tests rely on, so hard mode is passable from this page alone. Anything not listed is your
call.

**Conventions.** Every request passes `timeout=self.timeout` per request and sends
`User-Agent: self.user_agent`. Non-2xx → `APIError` with `.status_code`, `.body` (contains the
response text) and the status code in `str(exc)`. A 2xx body that isn't JSON, isn't the
documented shape, or has a required field missing or of the wrong type → `ResponseFormatError`
(never a bare `KeyError`/`TypeError`/`AttributeError`). Defaults: `DEFAULT_WIKIPEDIA_BASE_URL`,
`DEFAULT_MISTRAL_BASE_URL`, `DEFAULT_MODEL`; a trailing `/` on `base_url` is ignored.

**Stage 1.**
- `strip_html`: remove tags, *then* unescape entities (so `&lt;b&gt;` stays as the text `<b>`),
  collapse whitespace runs to one space, strip the ends. `""` → `""`.
- `search(query, limit)`: `ValueError` and no request if `query` is blank or `limit` is outside
  1..100. `GET {base_url}/w/rest.php/v1/search/page` with params `q=<query stripped>`,
  `limit=<limit>` and `Accept: application/json`. Each item of `pages` becomes
  `Document(title, key, url=f"{base_url}/wiki/{urllib.parse.quote(key)}", text=strip_html(excerpt),
  description)`, in order; missing/null `excerpt` or `description` → `""`; missing/non-str `key`
  or `title` → `ResponseFormatError`, as is a body that isn't a dict with a list `pages`.
  A 404 is not retried.

**Stage 2.**
- `chat`: `ConfigurationError` mentioning `MISTRAL_API_KEY` before any request when the key
  (argument, else the env var, read lazily so construction never fails) is missing or empty;
  empty `messages` → `ValueError`, no request. `POST {base_url}/chat/completions` with
  `Authorization: Bearer <key>`, `Accept: application/json`, `Content-Type: application/json`, and
  a JSON body of exactly `{"model", "messages"}` plus `temperature` / `max_tokens` only when they
  aren't `None`.
- Result: `ChatResult(id, model, content=choices[0].message.content, finish_reason, usage=Usage(...))`.
  `ResponseFormatError` for: non-JSON, `choices` missing or empty, `message` missing, `content`
  not a str, `usage` missing, a usage count that isn't an int. A 400 is not retried.

**Stage 3.**
- `build_context(docs, max_chars)`: block *n* (1-based) is `f"[{n}] {title}
{text}"`, blocks
  joined by `"

"`, total `len(context) <= max_chars`; whole documents only, stop at the first
  that doesn't fit; one `Source(n, title, url)` per block; nothing fits → `("", [])`.
- `build_messages(question, context)`: `[system, user]`. The system message, lower-cased,
  contains `only`, `sources`, `cite`, `[1]` and `i don't know`, and never the context. The user
  message contains `context` verbatim and `question`.
- `extract_citations(text, n)`: the `[k]` numbers, sorted, unique, `1 <= k <= n`.
- `answer(question)`: blank → `ValueError`. Calls `retriever.search(question, limit=top_k)`, then
  `llm.chat(build_messages(question, build_context(docs, max_context_chars)[0]),
  temperature=..., max_tokens=...)` with the pipeline's settings. Returns
  `Answer(content, sources, cited, usage, finish_reason)`. No sources → `Answer(NO_ANSWER, [],
  [], None)` without calling the LLM. `finish_reason` `"length"`/`"model_length"` → a
  `logging` warning; `"stop"` → no warning.

**Stage 4.**
- `parse_retry_after(value, now=...)`: seconds (`" 1.5 "` → `1.5`, negative → `0.0`), or an HTTP-date
  relative to `now` (clamped at 0; no timezone means UTC); `None`/unparseable → `None`.
- `compute_backoff(attempt, base, cap, rng)` = `rng() * min(cap, base * 2**attempt)`, and never
  overflows for huge `attempt`.
- `retry_delay(response, attempt, ...)`: a parseable `Retry-After` → `min(it, max_retry_after)`,
  no jitter; otherwise (or `response=None`) `compute_backoff(attempt, ...)`.
- `send_with_retries` / `async_send_with_retries` (same contract; the async one awaits `send`
  and `sleep`): at most `max_retries + 1` calls. Transient = status 408/429/500/502/503/504, or
  `send()` raising `httpx.TimeoutException`, `httpx.NetworkError` or `httpx.RemoteProtocolError`;
  before the next attempt sleep `retry_delay(response or None, attempt, ...)` where `attempt` is
  the 0-based index of the attempt that failed. 400/401/403/404/422 → plain `APIError` (not
  `RetryableError`), one call, no sleep. Out of attempts → `RetryableError(status_code=<last
  status or None>, body, attempts=max_retries + 1)`, `__cause__` set to the last exception, no
  sleep after the last attempt. `Retry-After` > `max_retry_after` → `RetryableError` at once,
  no sleep, `attempts` = calls made so far.
- Both sync clients retry with their own `max_retries`, `sleep`, `rng` and the default backoff
  parameters (base 0.5, cap 8, max Retry-After 60).
- `AsyncWikipediaClient.summary(key)`: `GET {base_url}/api/rest_v1/page/summary/{quote(key,
  safe="")}` (so `C++` → `C%2B%2B`), following redirects, retried like above. Body
  `{"title", "extract", "description"?, "content_urls": {"desktop": {"page"}}?}` →
  `Document(title, key, url=<desktop page, else f"{base_url}/wiki/{quote(key)}" whenever that
  nested value is missing or the wrong type>, text=extract, description or "")`; non-JSON, not a
  dict, or `title`/`extract` not a str → `ResponseFormatError`.
- `enrich(docs, max_concurrency)`: `ValueError` if `max_concurrency < 1`; exactly up to
  `max_concurrency` requests in flight (they must overlap); output in input order; a doc whose
  summary raises a `RAGError` is kept as-is.
- `parse_sse(lines)`: only `data:` lines count; `data: [DONE]` stops (don't read further);
  yield each non-empty `choices[0].delta.content`; a chunk that isn't JSON →
  `ResponseFormatError`.
- `stream(messages)`: the key and empty-`messages` checks raise when `stream()` is *called*, not
  on first iteration. Same URL, auth and timeout as `chat`, `Accept: text/event-stream`, body
  `{"model", "messages", "stream": true}`; yields the deltas; non-2xx → `APIError` with status
  and body.

## How to run

```bash
# all tests (offline: httpx.MockTransport fakes every response, no key or network needed)
uv run pytest interview/i01_mistral_rag -rA

# one stage at a time
uv run pytest interview/i01_mistral_rag -rA -k stage1

# check the tests against the reference solution
uv run pytest interview/i01_mistral_rag -rA --solution

# live: real Wikipedia; real Mistral if MISTRAL_API_KEY is set, otherwise an offline fake LLM
uv run python interview/i01_mistral_rag/run_live.py "What is retrieval-augmented generation?"
uv run python interview/i01_mistral_rag/run_live.py --solution --summaries "Who designed the Eiffel Tower?"
```

The tests inject an `httpx.Client` built on `httpx.MockTransport` and a `sleep` function that
only records durations, so retries are checked without any waiting. That's why every client
takes `client=`, `sleep=` and `rng=`. Without a key, writing the code this way is how you prove
it works.

Self-review afterwards: `solution/solution.py` (reference) and `solution/NOTES.md` (rubric,
common mistakes, follow-up questions, how to narrate while pairing). Don't open them before
you've tried.
