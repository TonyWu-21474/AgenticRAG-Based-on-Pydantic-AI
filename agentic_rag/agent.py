from __future__ import annotations

import json
import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from .config import Settings
from .corpus import Corpus, grep_corpus
from .embeddings import EmbeddingBackend
from .store import FaissStore
from .web import WebSearcher, dump

INSTRUCTIONS = """
You are a retrieval-augmented research agent. Answer from retrieved evidence, never from memory alone.

Tools:
- vector_search: semantic search over the local index. Start here for concepts, summaries and
  "how does X work" questions.
- grep_search: exact regular-expression search over the local files. Use it for identifiers,
  version strings, error messages, config keys, or to pin down an exact line.
- web_search: live web results. Use it for anything the local corpus does not cover, or for
  facts that change over time, then prefer primary sources.

Working style:
- Run searches before answering. Combine tools when one is not enough.
- Cite every claim inline as [source: path] for local files or [url] for web pages.
- If the evidence is missing or contradictory, say so plainly instead of guessing.
""".strip()


@dataclass
class RagDeps:
    """Everything the tools need, injected through RunContext."""

    corpus: Corpus
    store: FaissStore
    embedder: EmbeddingBackend
    web: WebSearcher
    settings: Settings


DEFAULT_MODEL = "ollama:qwen3:4b"


def register_tools(target: Agent[RagDeps, str]) -> None:
    @target.tool
    async def vector_search(ctx: RunContext[RagDeps], query: str, top_k: int = 6) -> str:
        """Semantic search over the local document index.

        Args:
            query: Natural-language question or topic to look for.
            top_k: Number of chunks to return, 1-20.
        """
        k = max(1, min(top_k, 20))
        hits = await ctx.deps.store.search(ctx.deps.embedder, query, k=k)
        if not hits:
            return "No matches in the local index."
        return json.dumps(
            [
                {
                    "source": chunk.source,
                    "chars": [chunk.start, chunk.end],
                    "score": round(score, 4),
                    "text": chunk.text,
                }
                for chunk, score in hits
            ],
            ensure_ascii=False,
        )

    @target.tool
    def grep_search(
        ctx: RunContext[RagDeps],
        pattern: str,
        glob: str = "",
        max_matches: int = 20,
        ignore_case: bool = True,
    ) -> str:
        """Regex search over local corpus files, returning matching lines with context.

        Args:
            pattern: Python regular expression, for example "class \\w+Agent" or "faiss|FAISS".
            glob: Optional file filter such as "*.md" or "docs/*.py". Empty means every file.
            max_matches: Maximum number of matching lines to return, 1-100.
            ignore_case: Match case-insensitively.
        """
        hits = grep_corpus(
            ctx.deps.corpus,
            pattern,
            glob=glob or None,
            max_matches=max(1, min(max_matches, 100)),
            ignore_case=ignore_case,
        )
        if not hits:
            return "No matches."
        return json.dumps(hits, ensure_ascii=False)

    @target.tool
    async def web_search(
        ctx: RunContext[RagDeps],
        query: str,
        max_results: int = 5,
        fetch_top: int = 2,
    ) -> str:
        """Search the live web for facts the local corpus does not cover.

        Args:
            query: Search query.
            max_results: Number of results to return, 1-10.
            fetch_top: How many of the top results to download in full, so you can read
                the page instead of only its snippet. 0 returns snippets only, max 5.
        """
        results = await ctx.deps.web.search(
            query,
            max_results=max(1, min(max_results, 10)),
            fetch_top=max(0, min(fetch_top, 5)),
        )
        if not results:
            return "No web results."
        return dump(results)


def build_agent(model: str | None = None) -> Agent[RagDeps, str]:
    """Create an agent with the three retrieval tools registered."""
    created: Agent[RagDeps, str] = Agent(
        model or os.getenv("RAG_MODEL", DEFAULT_MODEL),
        deps_type=RagDeps,
        instructions=INSTRUCTIONS,
    )
    register_tools(created)
    return created


agent = build_agent()


def registered_tools(target: Agent[RagDeps, str] | None = None) -> dict[str, object]:
    """The tools attached to `target` (defaults to the module-level agent)."""
    tools: dict[str, object] = {}
    for toolset in (target or agent).toolsets:
        tools.update(getattr(toolset, "tools", None) or {})
    return tools
