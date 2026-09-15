"""Agentic RAG built on Pydantic AI, with FAISS-backed vector search."""

from .agent import RagDeps, agent
from .config import Settings
from .corpus import Chunk, Corpus, grep_corpus
from .embeddings import EmbeddingBackend
from .store import FaissStore
from .web import WebResult, WebSearcher

__all__ = [
    "Chunk",
    "Corpus",
    "EmbeddingBackend",
    "FaissStore",
    "RagDeps",
    "Settings",
    "WebResult",
    "WebSearcher",
    "agent",
    "grep_corpus",
]
