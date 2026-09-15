from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".csv",
    ".html",
}


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a corpus file."""

    chunk_id: int
    source: str  # path relative to the corpus root, posix separators
    start: int  # character offset of the slice inside the file
    end: int
    text: str


def chunk_text(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Split `text` into ~`size` character windows, preferring line breaks."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size - 1))
    out: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if cut > size // 2:
                end = start + cut + 1
        out.append((start, end, text[start:end]))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


@dataclass
class Corpus:
    """Files on disk that the retrieval tools can read."""

    root: Path
    files: list[Path]
    _cache: dict[Path, str] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, root: Path) -> "Corpus":
        if not root.is_dir():
            raise FileNotFoundError(f"corpus directory not found: {root}")
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
        )
        return cls(root=root, files=files)

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def read(self, path: Path) -> str:
        cached = self._cache.get(path)
        if cached is None:
            cached = path.read_text(encoding="utf-8", errors="replace")
            self._cache[path] = cached
        return cached

    def chunks(self, size: int, overlap: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in self.files:
            source = self.relative(path)
            for start, end, text in chunk_text(self.read(path), size, overlap):
                if text.strip():
                    chunks.append(Chunk(len(chunks), source, start, end, text))
        return chunks


def grep_corpus(
    corpus: Corpus,
    pattern: str,
    *,
    glob: str | None = None,
    max_matches: int = 40,
    context_lines: int = 1,
    ignore_case: bool = True,
) -> list[dict]:
    """Regex search across corpus files, returning matching lines with context."""
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc

    files = corpus.files
    if glob:
        files = [p for p in files if fnmatch.fnmatch(corpus.relative(p), glob)]

    hits: list[dict] = []
    for path in files:
        lines = corpus.read(path).splitlines()
        for lineno, line in enumerate(lines, 1):
            if regex.search(line):
                lo = max(0, lineno - 1 - context_lines)
                hi = min(len(lines), lineno + context_lines)
                hits.append(
                    {
                        "source": corpus.relative(path),
                        "line": lineno,
                        "text": line.strip()[:400],
                        "context": "\n".join(lines[lo:hi])[:1200],
                    }
                )
                if len(hits) >= max_matches:
                    return hits
    return hits
