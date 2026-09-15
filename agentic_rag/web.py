from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from html import unescape

import httpx

_DROP_RE = re.compile(
    r"<(script|style|noscript|template|svg)\b.*?</\1>", re.IGNORECASE | re.DOTALL
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_RE = re.compile(
    r"</(p|div|li|tr|h[1-6]|section|article|header|footer|blockquote)\s*>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n\s*\n\s*\n+")


def html_to_text(html: str, limit: int = 1500) -> str:
    """Reduce an HTML document to readable text, capped at `limit` characters."""
    text = _COMMENT_RE.sub(" ", html)
    text = _DROP_RE.sub(" ", text)
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text)
    text = _SPACE_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()[:limit]


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    content: str = ""  # fetched page text, filled in for the top results


class WebSearcher:
    """Web search with a keyless default.

    Tavily or Brave are used when their API keys are present; otherwise the
    search falls back to DuckDuckGo. The top `fetch_top` results are downloaded
    and reduced to plain text, because search snippets alone are usually too
    thin to answer a question from.
    """

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout
        self._user_agent = "Mozilla/5.0 (compatible; agentic-rag/0.1)"

    async def search(
        self, query: str, max_results: int = 5, fetch_top: int = 2
    ) -> list[WebResult]:
        if key := os.getenv("TAVILY_API_KEY"):
            results = await self._tavily(query, max_results, key)
        elif key := os.getenv("BRAVE_API_KEY"):
            results = await self._brave(query, max_results, key)
        else:
            results = await self._duckduckgo(query, max_results)
        if fetch_top > 0 and results:
            results = await self._with_page_text(results, fetch_top)
        return results

    async def fetch_text(self, url: str, limit: int = 1500) -> str:
        """Download `url` and return its readable text, or "" on any failure."""
        if not url.startswith(("http://", "https://")):
            return ""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": self._user_agent},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except Exception:
            return ""
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return ""
        return html_to_text(resp.text, limit=limit)

    async def _with_page_text(
        self, results: list[WebResult], fetch_top: int
    ) -> list[WebResult]:
        head = results[:fetch_top]
        bodies = await asyncio.gather(*(self.fetch_text(r.url) for r in head))
        enriched = [
            WebResult(r.title, r.url, r.snippet, content or r.snippet)
            for r, content in zip(head, bodies)
        ]
        return enriched + list(results[len(head) :])

    async def _tavily(self, query: str, n: int, key: str) -> list[WebResult]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": key,
                    "query": query,
                    "max_results": n,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            WebResult(r.get("title", ""), r.get("url", ""), r.get("content", ""))
            for r in data.get("results", [])
        ]

    async def _brave(self, query: str, n: int, key: str) -> list[WebResult]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": n},
                headers={"Accept": "application/json", "X-Subscription-Token": key},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            WebResult(r.get("title", ""), r.get("url", ""), r.get("description", ""))
            for r in data.get("web", {}).get("results", [])
        ]

    async def _duckduckgo(self, query: str, n: int) -> list[WebResult]:
        from ddgs import DDGS

        def run() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=n))

        rows = await asyncio.to_thread(run)
        return [
            WebResult(r.get("title", ""), r.get("href", ""), r.get("body", ""))
            for r in rows
        ]


def dump(results: list[WebResult]) -> str:
    return json.dumps([asdict(r) for r in results], ensure_ascii=False)
