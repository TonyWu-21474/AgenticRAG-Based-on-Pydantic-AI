from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ollama exposes an OpenAI-compatible endpoint under /v1. Pydantic AI's
# `ollama:` provider reads this variable, so the default local setup needs no
# extra configuration.
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


def load_dotenv(path: Path | None = None) -> None:
    try:
        from dotenv import load_dotenv as _load
    except ImportError:  # python-dotenv is optional
        return
    _load(path or ROOT / ".env", override=False)


def prepare_environment() -> None:
    """Make `.env` and the local Ollama default visible to Pydantic AI.

    Runs at import time so that `Agent(...)` and `Embedder(...)` built at module
    scope already see the provider configuration.
    """
    load_dotenv()
    os.environ.setdefault("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


prepare_environment()


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    index_dir: Path
    model: str
    embed_model: str
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 6
    rebuild: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        prepare_environment()
        return cls(
            corpus_dir=Path(os.getenv("RAG_CORPUS", ROOT / "data" / "corpus")),
            index_dir=Path(os.getenv("RAG_INDEX", ROOT / "data" / "index")),
            model=os.getenv("RAG_MODEL", "ollama:qwen3:4b"),
            embed_model=os.getenv("RAG_EMBED_MODEL", "ollama:bge-m3"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("RAG_TOP_K", "6")),
            rebuild=os.getenv("RAG_REBUILD", "").lower() in {"1", "true", "yes"},
        )
