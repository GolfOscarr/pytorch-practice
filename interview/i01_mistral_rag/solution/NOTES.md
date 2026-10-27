# i01 self-review notes

Read these after the session. Compare your `rag.py` with `solution.py`, section by section.

## 1. Rubric: what an "API skills" pair-programming round looks for

| Signal | What strong looks like | Where in `solution.py` |
|---|---|---|
| Auth & secrets | Key read from env, never hard-coded or logged; a clear error when it's missing, raised *before* any request | `MistralClient.__init__`, `_headers` → `ConfigurationError` |
| Timeouts | Every request has an explicit timeout, passed per request so an injected client can't silently turn it off | `timeout=self.timeout` in every `get`/`post`/`stream` |
| Status handling | 2xx → parse; other 4xx → fail fast with status + body; 408/429/500/502/503/504 → retry | `send_with_retries`, `RETRYABLE_STATUS_CODES`, `_api_error` |
| Retries | Bounded attempts, exponential backoff, **jitter**, `Retry-After` honored, and a `Retry-After` longer than we'll wait fails fast instead of sleeping, no sleep after the last attempt, transport errors classified (timeout and stale keep-alive yes, bad URL no) | `compute_backoff`, `retry_delay`, `parse_retry_after`, `_retry_after_too_long`, `_TRANSIENT_EXCEPTIONS` |
| Response validation | Never `data["choices"][0]["message"]["content"]` unguarded: wrong shapes become one typed error with a useful message | `_json`, `_field`, `ResponseFormatError` |
| Etiquette | Descriptive `User-Agent`, `Accept` header, limit validated against the API's max, bounded concurrency | `WikipediaClient.search`, `AsyncWikipediaClient.enrich` |
| Clean interfaces | Typed dataclasses in and out, `Protocol`s for the retriever and LLM, prompt building pure and separate from I/O | `Document`/`ChatResult`/`Answer`, `Retriever`/`ChatModel`, `build_context`/`build_messages` |
| Testability | Clients, `sleep` and `rng` are injected, so everything runs offline in milliseconds | constructors; `test_rag.py` uses `httpx.MockTransport` |
| RAG judgment | Budget by whole documents in rank order, numbered citations mapped back and validated, no LLM call when there's nothing to ground on, retrieved text treated as untrusted, a truncated answer (`finish_reason` `length`) flagged, not passed off as complete | `build_context`, `extract_citations`, `RAGPipeline.answer`, `SYSTEM_PROMPT` |
| Async | `httpx.AsyncClient`, a semaphore per `enrich` call, `TaskGroup` (siblings cancelled on an unexpected error) with results read in input order, per-item failure degrades instead of failing the batch, no blocking calls | `AsyncWikipediaClient.enrich`, `async_send_with_retries` |
| Streaming | SSE parsed line by line, comments and keep-alives ignored, stops at `[DONE]`, arguments validated at call time (not on first `next()`), error status read before raising, no retry once tokens have gone out | `parse_sse`, `MistralClient.stream` |
| Resource hygiene | Close only clients you created; context managers | `_owns_client`, `close`/`aclose`, `__enter__`/`__aenter__` |

## 2. Common mistakes (and the test that catches each)

- **Hard-coded API key**, or reading `os.environ["MISTRAL_API_KEY"]` at import time (crashes on
  import with no key). → `test_stage2_api_key_read_from_environment`,
  `test_stage2_missing_api_key_raises_before_any_request`.
- **No timeout.** `httpx` defaults to 5 s, but `requests` defaults to *forever*, and an injected
  client may have `timeout=None`. → `test_stage1_search_sends_expected_request` checks the
  per-request timeout.
- **Retrying 400/401/404.** A malformed body or a bad key will never succeed; retrying just burns
  quota and delays the real error. → `test_stage4_client_errors_are_not_retried`,
  `test_stage4_mistral_client_does_not_retry_401`.
- **No backoff, or backoff without jitter.** Fixed `sleep(1)` loops hammer an overloaded server.
  Synchronized retries cause a thundering herd. → `test_stage4_503_retried_with_exponential_backoff_then_success`.
- **Ignoring `Retry-After`** on 429. The server tells you when to come back. → `test_stage4_429_retry_after_is_honored`.
- **Sleeping the cap on a huge `Retry-After`, then retrying.** If the server says "come back
  in an hour" and you'll wait 60 s at most, the retry is guaranteed to hit the same limit. Fail
  fast with a typed error. → `test_stage4_retry_after_longer_than_cap_gives_up_without_sleeping`.
- **Not retrying a stale keep-alive.** A pooled connection the server already closed raises
  `httpx.RemoteProtocolError: Server disconnected without sending a response`. It's not a
  `NetworkError`, so a narrow `except` misses it, yet a fresh connection usually works.
  → `test_stage4_stale_keepalive_disconnect_is_retried`.
- **Ignoring `finish_reason`.** `"length"` means the answer was cut off, maybe before its
  citations. Log it and surface it. → `test_stage3_pipeline_warns_and_exposes_truncated_answer`.
- **Sleeping after the final attempt.** It wastes up to `cap` seconds before raising anyway.
  → `test_stage4_max_retries_exhausted_raises_retryable_error`.
- **Swallowing errors** (`except Exception: return []`). An outage then looks like "no results",
  and the pipeline answers "I don't know" when the real problem is a 503. Catch narrowly and
  re-raise typed errors. Degrade only where it's a deliberate choice (one summary in `enrich`),
  and log it.
- **Not validating the response shape.** A proxy's HTML error page with status 200, or
  `content: null` on a tool call, becomes a `KeyError` deep in your code. → the
  `..._malformed_...` tests.
- **Mixing prompt building with I/O.** If `answer()` builds strings inline around HTTP calls, you
  can't unit-test the prompt or the budget. Keep `build_context`/`build_messages` pure.
- **Untestable code**: `httpx.get(...)` at module level, `time.sleep` hard-wired, randomness not
  injectable. Without a key (exactly the situation in the real interview) you then can't show
  anything works.
- **`print` instead of `logging`.** Libraries log; applications decide what to show.
- **Blocking calls in async code**: `time.sleep`, sync `httpx.Client`, or `requests` inside
  `async def` freezes every task on the loop. Also: an unbounded `gather` over 1000 URLs, or a
  semaphore that serializes everything (`Semaphore(1)`).
- **Budget that cuts mid-document**, or token budgeting that forgets the system prompt and the
  question.
- **Trusting citation numbers.** The model can cite `[7]` with three sources. Validate and drop.

## 3. Likely follow-up questions (with short model answers)

1. **How would you evaluate this RAG system?**
   Split it. *Retrieval*: build a labeled set of (question → relevant pages) and measure
   recall@k / MRR / nDCG. *Generation*: faithfulness/groundedness (is every claim supported by a
   cited source?), answer relevance, citation precision/recall, and "I don't know" accuracy on
   unanswerable questions. LLM-as-judge scales this, but judges are biased (verbosity, position,
   self-preference) and inconsistent. Calibrate against a human-labeled subset, use pairwise
   comparisons with randomized order, fix the judge model and prompt, and track inter-rater
   agreement. Keep a regression set in CI.
2. **Wikipedia excerpts are tiny. How would you chunk real documents?**
   Split by structure (headings, paragraphs) into roughly 200–500 token chunks with some overlap,
   and keep metadata (title, section, URL) for citations. Too small loses context, too large
   wastes budget and dilutes the embedding. Tune it on the retrieval metrics above, not by feel.
3. **Keyword search vs embeddings?**
   BM25/keyword is great for exact names, codes and rare terms. Embeddings (e.g. `mistral-embed`)
   handle paraphrase and semantics. In production, use **hybrid** retrieval (both, merged with
   reciprocal rank fusion), then a **reranker** (cross-encoder or LLM) over the top 50 to pick
   the final 5. Query rewriting helps too: the user's question is often a bad search query.
4. **Prompt injection from retrieved text?**
   Retrieved pages are untrusted input. Put them in the user turn inside clear delimiters, say in
   the system prompt that instructions in sources must be ignored, never give the model tools or
   secrets that a source could trigger, and filter/escape delimiter look-alikes. It reduces the
   risk but doesn't remove it. Defend in depth at the action layer, not only in the prompt.
5. **Caching?**
   Cache Wikipedia search and summaries by normalized query/key with a TTL (they change slowly),
   using an HTTP cache that respects `ETag`/`Cache-Control`. Cache LLM answers only for
   deterministic settings (temperature 0) and exact prompt hashes. Keep the system prompt fixed
   so provider-side prompt caching can work.
6. **Rate limiting across many workers?**
   Per-process backoff doesn't coordinate. Use a shared token bucket (e.g. in Redis) sized to
   the provider's RPM/TPM limits, plus a queue with backpressure. Keep jittered retries as the
   last line of defense, and use a circuit breaker so a dead upstream fails fast instead of
   piling up retries.
7. **Cost and latency?**
   Tokens dominate cost, so a tight context budget, a smaller model for easy questions (routing),
   and `max_tokens` all help. Latency: run retrieval calls concurrently, stream the answer, reuse
   connections (one client, keep-alive), and set timeouts that match an end-to-end SLO. Measure
   p50/p95 per stage.
8. **Why stream, and what changes in the UX?**
   Time-to-first-token matters more than total time for perceived speed. Show sources first
   (they're known before generation), then stream text. Handle mid-stream disconnects (show what
   you have, offer a retry). Don't retry automatically once tokens are out, or the user sees
   duplicates.
9. **What do you do when you have no API key (like in the real interview)?**
   Code against an interface and inject the HTTP client. Write fakes/contract tests from the
   documented request and response schemas (`httpx.MockTransport`, recorded fixtures), and an
   offline fake LLM so the rest of the pipeline runs for real. Say this out loud early: it turns
   "I can't test" into "here's how I know it works."
10. **Is retrying a POST safe?**
    Chat completions have no side effects beyond cost, so retrying on 5xx or a timeout is fine,
    though you might pay twice. For side-effecting POSTs, use an idempotency key, or only retry
    when you know the request never reached the server (e.g. a connect error).
11. **How do you budget in tokens instead of characters?**
    Use the model's real tokenizer (Mistral publishes `mistral-common`). `chars / 4` is only a
    rough average for English prose and *under*estimates tokens for code, numbers, URLs and
    non-Latin scripts, so it can overflow the window. If you must estimate, `chars / 3` is a safer
    bound. The budget is `context_window - system_prompt - question - max_tokens(answer) -
    safety margin`.
12. **What would you log and monitor?**
    Per request: latency, status, retry count, tokens, model, and a request id (never the key or
    full user text, for PII). Dashboards for error rate by status, 429 rate, p95 latency, cost per
    answer, and "I don't know" rate (a spike means retrieval broke).
13. **Let the model decide when to search: how, with Mistral's tool (function) calling?**
    Send `tools=[{"type": "function", "function": {"name": "search_wikipedia", "description":
    "...", "parameters": <JSON Schema for {"query": str}>}}]` and `tool_choice`: `"auto"` (model
    decides), `"any"` (must call some tool) or `"none"`. When the reply has
    `finish_reason == "tool_calls"`, read `choices[0].message.tool_calls[i]`: `.id`,
    `.function.name` and `.function.arguments`, which is a **JSON string**, not an object, so
    `json.loads` it and validate it against your schema (the model can emit bad JSON, unknown
    names or missing fields; answer that with an error message as the tool result rather than
    crashing). Append the assistant message *as returned* (with its `tool_calls`), then one
    `{"role": "tool", "tool_call_id": <id>, "name": <name>, "content": <result as a string>}` per
    call, and call the API again. Loop until `finish_reason` is `"stop"`, with a hard bound on
    iterations (say 3–5) and on total tokens, so a model that keeps calling tools can't spin.
    Tradeoff: *model decides* saves retrieval latency and context on questions that need no
    lookup ("hi", arithmetic) and can issue better or multiple queries, but it may skip search
    and answer from memory (ungrounded), costs an extra round trip when it does search, and is
    harder to test and evaluate. *Always retrieve* (this exercise) is one round trip,
    deterministic and auditable. A common middle ground: always retrieve, and expose a tool only
    for follow-up searches.
14. **Why raw HTTP instead of the `mistralai` SDK?**
    Here the interviewer asked for it, and it makes every API concern visible. In production the
    SDK is a reasonable choice: `from mistralai import Mistral; client = Mistral(api_key=...)`,
    then `client.chat.complete(model=..., messages=...)`, `client.chat.stream(model=...,
    messages=...)` (iterate the events) and `client.embeddings.create(model="mistral-embed",
    inputs=[...])`. It gives typed models and configurable retries, at the price of a dependency, less
    control over timeouts/retry policy, and harder transport-level fakes (you mock the SDK
    instead of `httpx.MockTransport`). Either is defensible; say which you picked and why.
    Whichever you use, check the response shape against the model you call: newer and reasoning
    models can return `message.content` as a **list of typed chunks** (e.g. `{"type": "text",
    "text": ...}` next to thinking chunks) instead of a string. This solution treats a non-str
    `content` as `ResponseFormatError`; a production client would have to handle the list form
    for those models.
15. **Why create the semaphore inside `enrich` and not in `__init__`?**
    Not because of "the loop must be running": since Python 3.10, asyncio primitives bind to a
    loop lazily, the first time they have to wait. The real reason is reuse: a semaphore stored
    on the client binds to the first `asyncio.run()` loop that waits on it, and the next
    `asyncio.run()` with the same client fails with "bound to a different event loop" (e.g. a
    sync wrapper that calls `asyncio.run(client.enrich(...))` per question). A per-call semaphore also
    makes the limit per batch; if you need a global limit across concurrent `enrich` calls, own
    one long-lived loop and create the semaphore on it. (The same caveat applies to the
    `httpx.AsyncClient` itself: its pooled connections belong to one loop, so prefer one loop
    for the client's whole life.)

## 4. How to talk through the code while pairing

**Ask first (1–2 minutes).** Good clarifying questions:
- "Is there a key for Mistral? If not, I'll inject the HTTP client and fake responses so we can
  test everything offline."
- "Sync or async? How many concurrent users, roughly?"
- "Which Wikipedia endpoint: search, summary, or both? Any language other than English?"
- "What should happen when nothing relevant is found: an 'I don't know', or answer from model
  knowledge?"
- "How strict is the context budget? Characters, tokens, which model's window?"
- "Do you want citations inline like `[1]`, and should I validate them?"
- "How should errors reach the caller: exceptions, or a result object?"
- "Is it OK to use `httpx` directly rather than the SDK?" (they said raw HTTP here)

**Narrate decisions, not keystrokes.** "I'm passing the timeout per request because the client
might be injected without one." "I'm not retrying 401: the key is wrong, and a retry won't fix
it." "This shape check exists because a proxy can return HTML with status 200."

**Timebox and say so.** "Basic search first, then the LLM call, then I'll wire them up; retries
later if we have time." Get an end-to-end path working early, then harden it. A complete simple
version beats a perfect half.

**Test as you go.** Write or run one test per behavior. When you're stuck on the environment
(no key, no network), say what you'd verify and how, then fake it.

**Close the loop.** In the last 2 minutes, list what you'd add next: retries across workers,
evaluation, caching, hybrid retrieval. It shows you know the gap between the demo and
production.
