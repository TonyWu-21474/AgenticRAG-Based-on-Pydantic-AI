from __future__ import annotations

import json
import math
import re
import zlib
from pathlib import Path

import pytest

from agentic_rag.corpus import Corpus, chunk_text, grep_corpus
from agentic_rag.store import FaissStore

DOCS = {
    "alpha.md": (
        "# Alpha\n\n"
        "Vector search uses dense embeddings to find chunks by meaning rather than\n"
        "by exact wording.\n"
    ),
    "beta.md": (
        "# Beta\n\n"
        "The zebra protocol keeps a rolling checksum of every transmitted frame so\n"
        "that silent corruption is detected immediately.\n"
    ),
    "gamma.py": (
        "def rebuild():\n"
        "    index = 'IndexFlatIP'\n"
        "    return index\n"
    ),
}


class FakeEmbeddingBackend:
    """Deterministic bag-of-words embeddings, so plumbing tests need no network."""

    def __init__(self, model: str = "fake:test", dim: int = 64) -> None:
        self.model = model
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    async def embed_documents(self, texts) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    root = tmp_path / "corpus"
    root.mkdir()
    for name, body in DOCS.items():
        (root / name).write_text(body, encoding="utf-8")
    return Corpus.load(root)


def test_chunk_text_splits_with_overlap() -> None:
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(40))
    pieces = chunk_text(text, size=200, overlap=50)

    assert len(pieces) > 1
    assert pieces[0][0] == 0
    assert pieces[-1][1] == len(text)
    for (_, end, _), (next_start, _, _) in zip(pieces, pieces[1:]):
        assert next_start < end, "consecutive chunks must overlap"
        assert next_start > 0
    assert "".join(piece[2] for piece in pieces).count("line 0 ") == 1


def test_chunk_text_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", size=0, overlap=0)


def test_corpus_loads_only_text_files(corpus: Corpus) -> None:
    assert [corpus.relative(p) for p in corpus.files] == [
        "alpha.md",
        "beta.md",
        "gamma.py",
    ]
    assert len(corpus.chunks(size=200, overlap=20)) >= 3


def test_grep_search_returns_line_numbers_and_context(corpus: Corpus) -> None:
    hits = grep_corpus(corpus, r"IndexFlatIP")

    assert len(hits) == 1
    assert hits[0]["source"] == "gamma.py"
    assert hits[0]["line"] == 2
    assert "IndexFlatIP" in hits[0]["text"]
    assert "def rebuild" in hits[0]["context"]


def test_grep_search_honours_glob_and_case(corpus: Corpus) -> None:
    assert grep_corpus(corpus, "zebra", glob="*.md")
    assert grep_corpus(corpus, "zebra", glob="*.py") == []
    assert grep_corpus(corpus, "ZEBRA", ignore_case=True)
    assert grep_corpus(corpus, "ZEBRA", ignore_case=False) == []


def test_grep_search_caps_matches(corpus: Corpus) -> None:
    hits = grep_corpus(corpus, r"\w+", max_matches=2)
    assert len(hits) == 2


def test_grep_search_rejects_invalid_regex(corpus: Corpus) -> None:
    with pytest.raises(ValueError):
        grep_corpus(corpus, "([unclosed")


async def test_faiss_store_builds_saves_loads_and_searches(
    corpus: Corpus, tmp_path: Path
) -> None:
    backend = FakeEmbeddingBackend()
    store = await FaissStore.build(
        corpus.chunks(size=200, overlap=20), backend, backend.model
    )

    assert store.dim == backend.dim
    assert len(store.chunks) >= 3

    directory = tmp_path / "index"
    store.save(directory)
    reloaded = FaissStore.load(directory)
    assert reloaded.dim == store.dim
    assert reloaded.embed_model == backend.model
    assert [chunk.text for chunk in reloaded.chunks] == [
        chunk.text for chunk in store.chunks
    ]

    query = "The zebra protocol keeps a rolling checksum of every transmitted frame"
    hits = await reloaded.search(backend, query, k=3)

    assert hits, "search must return results"
    top_chunk, top_score = hits[0]
    assert top_chunk.source == "beta.md"
    assert top_score > 0.6
    assert hits == sorted(hits, key=lambda pair: pair[1], reverse=True)
    assert top_score > hits[1][1]

    # Searching with a chunk's own text must be its own nearest neighbour with
    # cosine similarity 1, which proves normalisation + inner product is wired up.
    exact = await reloaded.search(backend, top_chunk.text, k=1)
    assert exact[0][0].chunk_id == top_chunk.chunk_id
    assert exact[0][1] == pytest.approx(1.0, abs=1e-3)


async def test_faiss_store_respects_k_and_rejects_model_mismatch(
    corpus: Corpus,
) -> None:
    backend = FakeEmbeddingBackend()
    store = await FaissStore.build(
        corpus.chunks(size=200, overlap=20), backend, backend.model
    )

    assert len(await store.search(backend, "zebra", k=1)) == 1
    assert len(await store.search(backend, "zebra", k=99)) == len(store.chunks)

    other = FakeEmbeddingBackend(model="fake:other")
    with pytest.raises(RuntimeError, match="rebuild"):
        await store.search(other, "zebra")


async def test_faiss_store_rejects_empty_corpus() -> None:
    backend = FakeEmbeddingBackend()
    with pytest.raises(ValueError):
        await FaissStore.build([], backend, backend.model)


def test_agent_exposes_the_three_retrieval_tools() -> None:
    from agentic_rag.agent import registered_tools

    tools = registered_tools()
    assert set(tools) == {"vector_search", "grep_search", "web_search"}
    for name, tool in tools.items():
        assert tool.description, f"{name} needs a docstring for the model"
        assert "Args:" in (tool.function.__doc__ or ""), f"{name} needs Args docs"

    vector_params = tools["vector_search"].function_schema.json_schema
    assert set(vector_params["properties"]) == {"query", "top_k"}
    grep_params = tools["grep_search"].function_schema.json_schema
    assert {"pattern", "glob", "max_matches", "ignore_case"} <= set(
        grep_params["properties"]
    )
    assert grep_params["required"] == ["pattern"]


def test_vector_search_tool_returns_citable_json(corpus: Corpus) -> None:
    import asyncio
    from types import SimpleNamespace

    from agentic_rag.agent import registered_tools

    backend = FakeEmbeddingBackend()
    store = asyncio.run(
        FaissStore.build(corpus.chunks(size=200, overlap=20), backend, backend.model)
    )
    ctx = SimpleNamespace(
        deps=SimpleNamespace(corpus=corpus, store=store, embedder=backend)
    )

    payload = json.loads(
        asyncio.run(
            registered_tools()["vector_search"].function(ctx, query="zebra checksum", top_k=2)
        )
    )

    assert len(payload) == 2
    assert payload[0]["source"] == "beta.md"
    assert payload[0]["chars"][1] > payload[0]["chars"][0]
    assert isinstance(payload[0]["score"], float)


def test_echo_thinking_prints_reasoning_but_not_the_answer(capsys) -> None:
    import asyncio

    from pydantic_ai.messages import (
        PartDeltaEvent,
        PartEndEvent,
        PartStartEvent,
        TextPart,
        ThinkingPart,
        ThinkingPartDelta,
    )

    from agentic_rag.cli import echo_thinking

    async def events():
        yield PartStartEvent(index=0, part=ThinkingPart(content="Check the index"))
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" class first."))
        yield PartEndEvent(index=0, part=ThinkingPart(content="Check the index class first."))
        yield PartStartEvent(index=1, part=TextPart(content="It is IndexFlatIP."))
        yield PartEndEvent(index=1, part=TextPart(content="It is IndexFlatIP."))

    asyncio.run(echo_thinking(None, events()))
    out = capsys.readouterr().out

    assert "thinking> Check the index class first." in out
    assert "IndexFlatIP" not in out


def test_echo_thinking_stays_silent_without_thinking_parts(capsys) -> None:
    import asyncio

    from pydantic_ai.messages import PartEndEvent, PartStartEvent, TextPart

    from agentic_rag.cli import echo_thinking

    async def events():
        yield PartStartEvent(index=0, part=TextPart(content="Answer only."))
        yield PartEndEvent(index=0, part=TextPart(content="Answer only."))

    asyncio.run(echo_thinking(None, events()))

    assert capsys.readouterr().out == ""


def test_html_to_text_strips_markup_scripts_and_styles() -> None:
    from agentic_rag.web import html_to_text

    html = """
    <html><head><style>body{color:red}</style>
    <script>var secret = 'nope';</script></head>
    <body><h1>FAISS &amp; friends</h1>
    <p>Inner product equals cosine similarity<br>when vectors are normalised.</p>
    <!-- a comment -->
    <div><a href="/x">IndexFlatIP</a></div></body></html>
    """
    text = html_to_text(html)

    assert "FAISS & friends" in text
    assert "IndexFlatIP" in text
    assert "secret" not in text
    assert "color:red" not in text
    assert "a comment" not in text
    assert "<" not in text
    assert html_to_text(html, limit=20) == text[:20]


def test_web_search_result_carries_fetched_page_text(monkeypatch) -> None:
    import asyncio

    from agentic_rag.web import WebResult, WebSearcher

    searcher = WebSearcher()

    async def fake_ddg(query: str, n: int) -> list[WebResult]:
        return [WebResult("Doc", "https://example.com/a", "short snippet")]

    async def fake_fetch(url: str, limit: int = 1500) -> str:
        return "Long page body about the zebra protocol."

    monkeypatch.setattr(searcher, "_duckduckgo", fake_ddg)
    monkeypatch.setattr(searcher, "fetch_text", fake_fetch)

    results = asyncio.run(searcher.search("zebra", max_results=1, fetch_top=1))
    assert results[0].content.startswith("Long page body")

    snippet_only = asyncio.run(searcher.search("zebra", max_results=1, fetch_top=0))
    assert snippet_only[0].content == ""


def test_web_search_falls_back_to_snippet_when_fetch_fails(monkeypatch) -> None:
    import asyncio

    from agentic_rag.web import WebResult, WebSearcher

    searcher = WebSearcher()

    async def fake_ddg(query: str, n: int) -> list[WebResult]:
        return [WebResult("Doc", "https://example.com/a", "short snippet")]

    async def no_network(url: str, limit: int = 1500) -> str:
        return ""

    monkeypatch.setattr(searcher, "_duckduckgo", fake_ddg)
    monkeypatch.setattr(searcher, "fetch_text", no_network)

    results = asyncio.run(searcher.search("zebra", max_results=1, fetch_top=1))
    assert results[0].content == "short snippet"


@pytest.mark.integration
def test_live_ollama_embeddings_and_index_roundtrip(tmp_path: Path) -> None:
    import asyncio

    from agentic_rag.cli import build_deps
    from agentic_rag.config import Settings

    settings = Settings(
        corpus_dir=Path("data/corpus"),
        index_dir=tmp_path / "index",
        model="ollama:qwen3:4b",
        embed_model="ollama:bge-m3",
    )
    deps = asyncio.run(build_deps(settings))

    assert deps.store.dim == 1024
    assert len(deps.store.chunks) >= 4
    assert (tmp_path / "index" / "index.faiss").exists()

    hits = asyncio.run(
        deps.store.search(
            deps.embedder, "why is inner product equal to cosine similarity", k=3
        )
    )
    assert hits[0][0].source == "faiss_notes.md"
    assert hits[0][1] > 0.4


@pytest.mark.integration
def test_live_agent_calls_a_tool_and_cites_a_source() -> None:
    import asyncio

    from agentic_rag.agent import agent
    from agentic_rag.cli import build_deps, tools_used
    from agentic_rag.config import Settings

    async def run():
        deps = await build_deps(Settings.from_env())
        result = await agent.run(
            "Which FAISS index class does this project use, and which function "
            "normalises the vectors? Cite the file you used.",
            deps=deps,
        )
        return result, tools_used(result)

    result, used = asyncio.run(run())

    assert set(used) & {"vector_search", "grep_search"}, f"no retrieval tool used: {used}"
    assert "faiss_notes.md" in result.output
    assert "IndexFlatIP" in result.output
    assert "normalize_L2" in result.output


@pytest.mark.integration
def test_live_web_search_returns_results() -> None:
    import asyncio

    from agentic_rag.web import WebSearcher

    results = asyncio.run(
        WebSearcher().search("Pydantic AI documentation", max_results=3, fetch_top=2)
    )

    assert results
    for item in results:
        assert item.title
        assert item.url.startswith("http")
    assert any(len(item.content) > 200 for item in results), (
        "at least one result should carry real page text, not just a snippet"
    )
