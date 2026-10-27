"""Run the RAG pipeline against the real Wikipedia API (and Mistral, if a key is set).

    uv run python interview/i01_mistral_rag/run_live.py "What is retrieval-augmented generation?"
    uv run python interview/i01_mistral_rag/run_live.py --solution --summaries "question"

Uses your rag.py by default; --solution uses solution/solution.py. Without MISTRAL_API_KEY an
offline fake LLM stands in (it just lists the sources it was given), so the retrieval half still
runs live.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx

HERE = Path(__file__).parent


def load(use_solution: bool) -> ModuleType:
    path = HERE / "solution" / "solution.py" if use_solution else HERE / "rag.py"
    spec = importlib.util.spec_from_file_location("i01_rag", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered while executing
    spec.loader.exec_module(module)
    return module


def make_fake_llm(rag: ModuleType) -> Any:
    class EchoLLM:
        """Offline stand-in for MistralClient: reports which sources it would have used."""

        def chat(
            self,
            messages: Sequence[dict[str, str]],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> Any:
            user = messages[-1]["content"]
            headers = [line for line in user.splitlines() if line.startswith("[")]
            content = (
                "(offline fake LLM: MISTRAL_API_KEY is not set) "
                f"I was given {len(headers)} sources: " + "; ".join(headers)
            )
            n = len(user)
            return rag.ChatResult(
                id="fake",
                model="echo",
                content=content,
                finish_reason="stop",
                usage=rag.Usage(prompt_tokens=n // 4, completion_tokens=0, total_tokens=n // 4),
            )

    return EchoLLM()


def make_summary_retriever(rag: ModuleType, wiki: Any, max_concurrency: int) -> Any:
    class SummaryRetriever:
        """Search, then swap snippets for full page summaries fetched concurrently."""

        def search(self, query: str, limit: int = 5) -> list[Any]:
            docs = wiki.search(query, limit=limit)

            async def enrich() -> list[Any]:
                async with rag.AsyncWikipediaClient() as client:
                    result: list[Any] = await client.enrich(docs, max_concurrency=max_concurrency)
                    return result

            return asyncio.run(enrich())

    return SummaryRetriever()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("question")
    parser.add_argument("--solution", action="store_true", help="use solution/solution.py")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=4000, help="context budget")
    parser.add_argument(
        "--summaries", action="store_true", help="fetch full page summaries (async)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log retries")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )

    rag = load(args.solution)
    print(f"using {'solution/solution.py' if args.solution else 'rag.py'}")

    if os.environ.get("MISTRAL_API_KEY"):
        llm: Any = rag.MistralClient()
    else:
        print("MISTRAL_API_KEY is not set: using an offline fake LLM (Wikipedia is still live).")
        llm = make_fake_llm(rag)

    wiki = rag.WikipediaClient()
    retriever = make_summary_retriever(rag, wiki, 4) if args.summaries else wiki
    pipeline = rag.RAGPipeline(retriever, llm, top_k=args.top_k, max_context_chars=args.max_chars)
    try:
        answer = pipeline.answer(args.question)
    except NotImplementedError:
        print("rag.py is not implemented yet. Finish the stages, or run with --solution.")
        return 2
    except rag.ConfigurationError as exc:
        print(f"configuration error: {exc}")
        return 2
    except (rag.APIError, httpx.HTTPError) as exc:
        print(f"network/API error (offline or sandboxed?): {type(exc).__name__}: {exc}")
        return 1
    except rag.RAGError as exc:  # e.g. ResponseFormatError: the API answered something unexpected
        print(f"unexpected API response: {type(exc).__name__}: {exc}")
        return 1
    finally:
        wiki.close()
        if hasattr(llm, "close"):
            llm.close()

    print(f"\nQ: {args.question}\n\nA: {answer.text}\n")
    for source in answer.sources:
        mark = "*" if source.index in answer.cited else " "
        print(f" {mark}[{source.index}] {source.title} <{source.url}>")
    if answer.usage is not None:
        print(f"\nusage: {answer.usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
