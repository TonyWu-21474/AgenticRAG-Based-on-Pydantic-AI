# Agentic RAG architecture

This project is a retrieval-augmented generation agent built on the Pydantic AI
framework. The agent is deliberately thin: all knowledge comes from three
retrieval tools, and the model decides which one to call.

## Layers

1. `corpus.py` loads text files from `data/corpus`, splits them into overlapping
   chunks, and provides a regular-expression search used by `grep_search`.
2. `embeddings.py` wraps the Pydantic AI `Embedder`, so the chat model and the
   embedding model are both configured with Pydantic AI provider strings.
3. `store.py` builds and persists the FAISS index, storing chunk metadata next
   to the binary index so results can be cited back to a file and offset.
4. `agent.py` declares the agent, its instructions, and the three tools.

## Retrieval flow

A question arrives, the agent picks a tool, the tool result goes back into the
model context, and the loop repeats until the model answers. Because tool
results carry `source` plus character offsets, the final answer can cite exact
locations such as `[source: architecture.md]`.

## Tool selection guidance

- Semantic questions ("how does retrieval work here") start in `vector_search`.
- Identifiers and exact strings ("IndexFlatIP", "RAG_REBUILD") go to
  `grep_search`.
- Anything outside the corpus, or facts that change over time, goes to
  `web_search`.
