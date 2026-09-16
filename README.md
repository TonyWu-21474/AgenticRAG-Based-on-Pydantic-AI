# Agentic RAG with Pydantic AI and FAISS

A retrieval-augmented agent built on [Pydantic AI](https://pydantic.dev/docs/ai/overview/).
The agent owns no knowledge of its own: it answers from three retrieval tools and
cites what it used.

| Tool | Purpose |
| --- | --- |
| `vector_search` | Semantic search over the corpus, backed by a FAISS index |
| `grep_search` | Regex search over the corpus files, for exact strings and identifiers |
| `web_search` | Live web search, with the top pages downloaded and read |

The model decides which tool to call and may call several before answering. Tools
return `source` plus character offsets (`vector_search`) or `source` plus line
numbers (`grep_search`), so every claim can be traced back to a location.

## Layout

```
agentic_rag/
  config.py       environment-driven settings
  corpus.py       file discovery, chunking, regex search
  embeddings.py   wrapper around the Pydantic AI Embedder
  store.py        FAISS index build / save / load / search
  web.py          web search backends and page text extraction
  agent.py        agent, instructions, and the three tools
  cli.py          interactive chat loop
data/corpus/      your documents (markdown, text, python, ...)
data/index/       generated: index.faiss + chunks.json
tests/            unit tests plus opt-in integration tests
```

## Setup

```powershell
cd F:\PydanticAI
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to override the defaults. The default
configuration targets a local [Ollama](https://ollama.com) server, which needs no
API key:

```powershell
ollama pull qwen3:4b     # chat model with tool calling
ollama pull bge-m3       # embedding model (1024 dimensions)
```

To use hosted models instead, set `RAG_MODEL=openai:gpt-4o-mini`,
`RAG_EMBED_MODEL=openai:text-embedding-3-small`, and `OPENAI_API_KEY`.

## Run

```powershell
.\.venv\Scripts\python.exe -m agentic_rag.cli
```

Each turn prints the model's chain of thought as it streams, one `thinking> ...`
block per model request, and then the final answer as `agent> ...`, along with the
tool calls the run made. Ollama returns the reasoning trace in the `reasoning`
field for thinking models such as `qwen3:4b`; Pydantic AI maps that to a
`ThinkingPart`, and the CLI forwards those parts to the terminal while the model
is still generating. A model without the `thinking` capability simply produces no
`thinking>` block.

The index is built on first run and reused afterwards. Put documents in
`data/corpus/`, then rebuild after editing them:

```powershell
$env:RAG_REBUILD = "1"; .\.venv\Scripts\python.exe -m agentic_rag.cli
```

## Tools in detail

`vector_search(query, top_k=6)` embeds the query with the same model used for
indexing and searches a FAISS `IndexFlatIP`. Vectors are L2-normalised before
they are added and before every query, which turns the inner product into cosine
similarity. Results carry the source path, character offsets, the score, and the
chunk text.

`grep_search(pattern, glob="", max_matches=20, ignore_case=True)` compiles the
pattern as a Python regular expression and walks the corpus, returning matching
lines with one line of surrounding context. Use it for identifiers, config keys,
error strings, and anything where the exact wording matters.

`web_search(query, max_results=5, fetch_top=2)` searches Tavily or Brave when
`TAVILY_API_KEY` or `BRAVE_API_KEY` is set, and DuckDuckGo otherwise. The top
`fetch_top` results are downloaded and reduced to plain text, because snippets are
usually too thin to answer from. Fetched pages that fail fall back to their
snippet.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_MODEL` | `ollama:qwen3:4b` | chat model |
| `RAG_EMBED_MODEL` | `ollama:bge-m3` | embedding model |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | local Ollama endpoint |
| `RAG_CHUNK_SIZE` | `900` | characters per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | characters of overlap |
| `RAG_TOP_K` | `6` | default number of vector results |
| `RAG_REBUILD` | unset | set to `1` to rebuild the index on startup |

Changing `RAG_EMBED_MODEL` invalidates every stored vector, so the loader refuses
to mix models and tells you to rebuild instead of returning meaningless scores.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest              # unit tests, no network
.\.venv\Scripts\python.exe -m pytest -m integration   # needs Ollama + network
```

The unit tests run against a deterministic fake embedding backend, so they cover
chunking, chunk boundaries, regex search, glob filtering, index
save/load/search, score ordering, model-mismatch detection, HTML-to-text
extraction, and the tool schemas the model sees. The integration tests exercise
the real Ollama embedding model, a real FAISS index round trip, a real agent turn
that must call a retrieval tool and cite a source, and live web search.
