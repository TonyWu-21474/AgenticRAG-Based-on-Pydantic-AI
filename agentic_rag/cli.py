from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ThinkingPart,
    ThinkingPartDelta,
)

from .agent import RagDeps, agent
from .config import Settings
from .corpus import Corpus
from .embeddings import EmbeddingBackend
from .store import FaissStore
from .web import WebSearcher


async def build_deps(settings: Settings) -> RagDeps:
    corpus = Corpus.load(settings.corpus_dir)
    embedder = EmbeddingBackend(settings.embed_model)
    index_file = settings.index_dir / "index.faiss"

    if settings.rebuild or not index_file.exists():
        store = await FaissStore.build(
            corpus.chunks(settings.chunk_size, settings.chunk_overlap),
            embedder,
            settings.embed_model,
        )
        store.save(settings.index_dir)
    else:
        store = FaissStore.load(settings.index_dir)

    if store.embed_model != embedder.model:
        raise RuntimeError(
            f"index was built with {store.embed_model!r} but the embedder is "
            f"{embedder.model!r}; set RAG_REBUILD=1 to rebuild"
        )
    return RagDeps(corpus, store, embedder, WebSearcher(), settings)


def describe(deps: RagDeps, settings: Settings) -> str:
    return (
        f"model={settings.model} embeddings={settings.embed_model} "
        f"files={len(deps.corpus.files)} chunks={len(deps.store.chunks)} dim={deps.store.dim}"
    )


def tools_used(result) -> list[str]:
    names: list[str] = []
    for message in result.all_messages():
        for part in getattr(message, "parts", []):
            name = getattr(part, "tool_name", None)
            if name and name not in names:
                names.append(name)
    return names


async def echo_thinking(ctx: RunContext[RagDeps], events: AsyncIterator[AgentStreamEvent]) -> None:
    """Print the model's chain of thought as it streams, one block per model request."""
    open_block = False
    async for event in events:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            if open_block:
                print(flush=True)
            print("thinking> ", end="", flush=True)
            open_block = True
            if event.part.content:
                print(event.part.content, end="", flush=True)
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
            if not open_block:
                print("thinking> ", end="", flush=True)
                open_block = True
            if event.delta.content_delta:
                print(event.delta.content_delta, end="", flush=True)
        elif isinstance(event, PartEndEvent) and isinstance(event.part, ThinkingPart):
            print(flush=True)
            open_block = False
    if open_block:
        print(flush=True)


async def main() -> None:
    settings = Settings.from_env()
    deps = await build_deps(settings)
    print(f"ready: {describe(deps, settings)}")
    print("ask a question, or type exit to quit.")

    history = None
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        result = await agent.run(
            question,
            deps=deps,
            message_history=history,
            event_stream_handler=echo_thinking,
        )
        history = result.new_messages()
        used = tools_used(result)
        if used:
            print(f"[tools: {', '.join(used)}]")
        print(f"\nagent> {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
