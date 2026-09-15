from __future__ import annotations

from typing import Sequence


class EmbeddingBackend:
    """Thin wrapper over Pydantic AI's Embedder.

    Keeping this behind a small interface means the FAISS layer never depends on
    a specific provider, and tests can swap in `TestEmbeddingModel`.
    """

    def __init__(self, model: str) -> None:
        from pydantic_ai import Embedder

        self.model = model
        self._embedder = Embedder(model)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        result = await self._embedder.embed_documents(list(texts))
        return result.embeddings

    async def embed_query(self, text: str) -> list[float]:
        result = await self._embedder.embed_query(text)
        return result.embeddings[0]
