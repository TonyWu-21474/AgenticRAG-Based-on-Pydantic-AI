# Operations

## Configuration

Settings come from environment variables, optionally seeded from a `.env` file.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_MODEL` | `ollama:qwen3:4b` | chat model |
| `RAG_EMBED_MODEL` | `ollama:bge-m3` | embedding model |
| `RAG_CHUNK_SIZE` | `900` | characters per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | characters of overlap |
| `RAG_TOP_K` | `6` | default vector results |
| `RAG_REBUILD` | unset | set to `1` to rebuild the index |

## Rebuilding the index

The index is built automatically the first time the CLI runs and reused on
later runs. Rebuild after editing the corpus, after changing `RAG_CHUNK_SIZE`,
or after switching `RAG_EMBED_MODEL`. Changing the embedding model invalidates
every stored vector, which is why the loader refuses to mix models and instead
raises an error telling you to rebuild.

## Failure modes

- A missing `data/corpus` directory raises `FileNotFoundError` at startup.
- An embedding-model mismatch raises `RuntimeError` rather than returning
  meaningless similarities.
- If the web search backend is unreachable, `web_search` returns no results and
  the agent is instructed to say the evidence is missing.
