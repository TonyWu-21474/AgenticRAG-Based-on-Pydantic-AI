from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .corpus import Chunk
from .embeddings import EmbeddingBackend

INDEX_FILE = "index.faiss"
META_FILE = "chunks.json"


class FaissStore:
    """FAISS vector index plus the chunk metadata needed to cite results.

    Vectors are L2-normalised and stored in an inner-product index, which makes
    the search score cosine similarity.
    """

    def __init__(self, index: faiss.Index, chunks: list[Chunk], embed_model: str) -> None:
        self.index = index
        self.chunks = chunks
        self.embed_model = embed_model

    @property
    def dim(self) -> int:
        return int(self.index.d)

    @classmethod
    async def build(
        cls,
        chunks: list[Chunk],
        embedder: EmbeddingBackend,
        embed_model: str,
        batch_size: int = 32,
    ) -> "FaissStore":
        if not chunks:
            raise ValueError("corpus produced no chunks; add documents first")
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = [c.text for c in chunks[i : i + batch_size]]
            vectors.extend(await embedder.embed_documents(batch))
        matrix = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index=index, chunks=list(chunks), embed_model=embed_model)

    async def search(
        self, embedder: EmbeddingBackend, query: str, k: int = 6
    ) -> list[tuple[Chunk, float]]:
        if self.embed_model != embedder.model:
            raise RuntimeError(
                f"index was built with {self.embed_model!r} but the embedder is "
                f"{embedder.model!r}; set RAG_REBUILD=1 to rebuild"
            )
        vector = np.ascontiguousarray(
            np.asarray([await embedder.embed_query(query)], dtype="float32")
        )
        faiss.normalize_L2(vector)
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))
        return [
            (self.chunks[int(idx)], float(score))
            for score, idx in zip(scores[0], ids[0])
            if int(idx) != -1
        ]

    def save(self, directory: Path) -> None:
        """Persist the index and its chunk metadata.

        The index goes through `faiss.serialize_index` + `Path.write_bytes`
        instead of `faiss.write_index`, because FAISS's own file writer cannot
        open paths containing non-ASCII characters on Windows (for example a
        user profile directory such as `C:\\Users\\<non-ascii>`). The bytes are
        the same format, so `faiss.read_index` can still read the file.
        """
        directory.mkdir(parents=True, exist_ok=True)
        (directory / INDEX_FILE).write_bytes(faiss.serialize_index(self.index).tobytes())
        payload = {
            "embed_model": self.embed_model,
            "dim": self.dim,
            "chunks": [asdict(c) for c in self.chunks],
        }
        (directory / META_FILE).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "FaissStore":
        blob = np.frombuffer((directory / INDEX_FILE).read_bytes(), dtype="uint8").copy()
        index = faiss.deserialize_index(blob)
        payload = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk(**row) for row in payload["chunks"]]
        return cls(index=index, chunks=chunks, embed_model=payload["embed_model"])
