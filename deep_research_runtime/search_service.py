"""Dual-layer search service for Deep Research.

General search engines provide broad web discovery. Vertical plugins add
domain-specific evidence only when a task asks for it, such as academic papers.
All provider-specific payloads are normalized into the shared Document shape.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Protocol
from urllib.parse import urlparse

import aiohttp
try:
    from ddgs import DDGS
except Exception:  # pragma: no cover - optional fallback dependency
    DDGS = None  # type: ignore
from lxml import html as lxml_html
from readability import Document as ReadabilityDocument

from .models import Document
from .settings import Settings

logger = logging.getLogger("DeepResearchMCP")


class Retriever(Protocol):
    name: str
    source_layer: str

    async def search(
        self, session: aiohttp.ClientSession, query: str, profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        ...


class _CircuitBreaker:
    """Per-engine failure tracker.

    States are implicit and derived from two counters:

    * ``consecutive_failures``: how many calls in a row raised or returned no
      results due to a remote error.
    * ``opened_at``: timestamp when we last tripped open. While
      ``now < opened_at + cooldown`` we say the breaker is OPEN and skip the
      retriever entirely.

    The breaker is intentionally minimal — no half-open state, no fancy stats —
    because the failure modes we want to cover (a single engine 5xx-ing for a
    few minutes) only need a coarse "skip me for 60s" semantic.
    """

    __slots__ = ("consecutive_failures", "opened_at", "threshold", "cooldown")

    def __init__(self, threshold: int, cooldown: float) -> None:
        self.consecutive_failures = 0
        self.opened_at = 0.0
        self.threshold = max(0, int(threshold))
        self.cooldown = max(0.0, float(cooldown))

    def is_open(self, now: float) -> bool:
        if self.threshold <= 0:
            return False
        if self.consecutive_failures < self.threshold:
            return False
        return (now - self.opened_at) < self.cooldown

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = 0.0

    def record_failure(self, now: float) -> None:
        self.consecutive_failures += 1
        if self.threshold > 0 and self.consecutive_failures >= self.threshold:
            self.opened_at = now


class SearchService:
    """Search facade with a general layer and a vertical layer."""

    def __init__(
        self,
        settings: Settings,
        *,
        general_retrievers: List[Retriever] | None = None,
        vertical_retrievers: Dict[str, List[Retriever]] | None = None,
    ) -> None:
        self.settings = settings
        self.general_retrievers = general_retrievers if general_retrievers is not None else self._build_general_retrievers()
        self.vertical_retrievers = vertical_retrievers if vertical_retrievers is not None else self._build_vertical_retrievers()
        # Per-engine circuit breakers and rate limiters. Created lazily so a
        # caller passing custom retrievers (e.g. test doubles) doesn't need to
        # know about either mechanism.
        self._breakers: Dict[str, _CircuitBreaker] = {}
        self._rate_limits: Dict[str, asyncio.Semaphore] = {}

    def _breaker_for(self, name: str) -> _CircuitBreaker:
        breaker = self._breakers.get(name)
        if breaker is None:
            breaker = _CircuitBreaker(
                threshold=self.settings.search_circuit_breaker_threshold,
                cooldown=self.settings.search_circuit_breaker_cooldown_sec,
            )
            self._breakers[name] = breaker
        return breaker

    def _rate_limiter_for(self, name: str) -> asyncio.Semaphore | None:
        limit = int(self.settings.search_engine_rate_limit or 0)
        if limit <= 0:
            return None
        sem = self._rate_limits.get(name)
        if sem is None:
            sem = asyncio.Semaphore(limit)
            self._rate_limits[name] = sem
        return sem

    async def _call_retriever(
        self,
        retriever: Retriever,
        session: aiohttp.ClientSession,
        query: str,
        profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run a single retriever subject to the circuit breaker + rate limit.

        Returns an empty list (rather than raising) when the breaker is open or
        when the call fails. Failures are recorded so repeated transients trip
        the breaker; a successful call resets it. The caller (``_run_retrievers``)
        treats empty lists as "this engine contributed nothing", which is the
        exact behavior we want when an engine is being skipped.
        """
        name = getattr(retriever, "name", retriever.__class__.__name__)
        now = time.monotonic()
        breaker = self._breaker_for(name)
        if breaker.is_open(now):
            # Quiet at info level — we don't want to spam logs every search.
            logger.debug("Skipping retriever %s: circuit breaker open", name)
            return []
        sem = self._rate_limiter_for(name)
        try:
            if sem is None:
                result = await retriever.search(session, query, profile)
            else:
                async with sem:
                    result = await retriever.search(session, query, profile)
        except Exception as exc:
            breaker.record_failure(time.monotonic())
            logger.warning("Retriever %s raised: %s", name, exc)
            return []
        breaker.record_success()
        return list(result or [])

    def _build_general_retrievers(self) -> List[Retriever]:
        engines = self.settings.get_active_search_engines()
        retrievers: List[Retriever] = []
        for engine in engines:
            if engine == "tavily":
                retrievers.append(TavilyRetriever(self.settings))
            elif engine == "exa":
                retrievers.append(ExaRetriever(self.settings))
            elif engine == "serper":
                retrievers.append(SerperRetriever(self.settings))
            elif engine == "bocha":
                retrievers.append(BochaRetriever(self.settings))
            elif engine == "serpapi":
                retrievers.append(SerpApiRetriever(self.settings))
            elif engine == "bing":
                retrievers.append(BingRetriever(self.settings))
            elif engine == "google":
                retrievers.append(GoogleRetriever(self.settings))
            elif engine == "searx":
                retrievers.append(SearxRetriever(self.settings))
        if self.settings.enable_duckduckgo_fallback:
            retrievers.append(DuckDuckGoRetriever(self.settings))
        return retrievers

    def _build_vertical_retrievers(self) -> Dict[str, List[Retriever]]:
        retrievers: Dict[str, List[Retriever]] = {}
        if self.settings.enable_academic_search:
            retrievers["academic"] = [SemanticScholarRetriever(self.settings)]
            if self.settings.enable_arxiv_search:
                retrievers["academic"].append(ArxivRetriever(self.settings))
            if self.settings.enable_pubmed_search:
                retrievers["academic"].append(PubMedCentralRetriever(self.settings))
        return retrievers

    @staticmethod
    def infer_verticals(text: str, explicit: List[str] | None = None) -> List[str]:
        values = [str(item).strip().lower() for item in (explicit or []) if str(item).strip()]
        lowered = str(text or "").lower()
        academic_terms = (
            "paper", "papers", "publication", "publications", "journal", "conference",
            "ieee", "acm", "arxiv", "pubmed", "semantic scholar", "citation",
            "论文", "期刊", "会议", "综述", "文献", "学术", "高被引",
        )
        if "academic" in values or any(term in lowered for term in academic_terms):
            values.append("academic")
        return _dedupe(values)

    async def reconnaissance(
        self,
        session: aiohttp.ClientSession,
        queries: List[str],
        *,
        verticals: List[str] | None = None,
        max_results_per_query: int = 3,
    ) -> List[Document]:
        profile = {
            "max_results": max(1, max_results_per_query),
            "include_raw_content": True,
            "search_depth": "basic",
            "topic": "general",
        }
        all_docs: List[Document] = []
        for query in queries:
            all_docs.extend(await self.search(session, query, profile, verticals=verticals))
        return self.dedupe_and_rank(all_docs)[: max(3, max_results_per_query * max(1, len(queries)))]

    async def search(
        self,
        session: aiohttp.ClientSession,
        query: str,
        profile: Dict[str, Any] | None = None,
        *,
        verticals: List[str] | None = None,
        source_types: List[str] | None = None,
    ) -> List[Document]:
        profile = dict(profile or {})
        explicit_verticals = list(verticals or [])
        if source_types:
            explicit_verticals.extend(source_types)
        active_verticals = self.infer_verticals(query, explicit_verticals)

        general_docs = await self._run_retrievers(session, self.general_retrievers, query, profile)
        vertical_docs: List[Document] = []
        for vertical in active_verticals:
            retrievers = self.vertical_retrievers.get(vertical, [])
            if retrievers:
                vertical_docs.extend(await self._run_retrievers(session, retrievers, query, profile))

        docs = await self._extract_missing_content(session, general_docs + vertical_docs)
        return self.dedupe_and_rank(docs)

    async def _run_retrievers(
        self,
        session: aiohttp.ClientSession,
        retrievers: List[Retriever],
        query: str,
        profile: Dict[str, Any],
    ) -> List[Document]:
        if not retrievers:
            return []
        # Each retriever call is wrapped by the circuit-breaker / rate-limiter
        # helper, so exceptions are absorbed and "skipped" engines just yield
        # empty results. We no longer need return_exceptions=True.
        results = await asyncio.gather(
            *(self._call_retriever(retriever, session, query, profile) for retriever in retrievers),
        )
        docs: List[Document] = []
        for retriever, result in zip(retrievers, results):
            for item in result:
                docs.append(self.normalize_document(item, source_name=retriever.name, source_layer=retriever.source_layer))
        return docs

    async def _extract_missing_content(
        self, session: aiohttp.ClientSession, documents: List[Document]
    ) -> List[Document]:
        if session is None:
            return documents
        semaphore = asyncio.Semaphore(max(1, self.settings.page_cleaning_concurrency))

        async def fill(document: Document) -> Document:
            async with semaphore:
                content = str(document.get("content") or "").strip()
                if content and len(content) >= 500:
                    return document
                url = str(document.get("url") or document.get("pdf_url") or "").strip()
                if not url:
                    return document
                extracted = await self.extract_url(session, url)
                if len(extracted) > len(content):
                    document = dict(document)
                    document["content"] = extracted
                    document["raw_content"] = extracted
                return document

        return await asyncio.gather(*(fill(doc) for doc in documents))

    async def extract_url(self, session: aiohttp.ClientSession, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return ""
        if url.lower().split("?", 1)[0].endswith(".pdf"):
            return await self._extract_pdf(session, url)
        if self.settings.jina_api_key or self.settings.use_jina_reader_without_key:
            text = await self._extract_jina(session, url)
            if text:
                return text
        return await self._extract_readability(session, url)

    async def _extract_jina(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            headers = {"Accept": "text/markdown"}
            if self.settings.jina_api_key:
                headers["Authorization"] = f"Bearer {self.settings.jina_api_key}"
            async with session.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.settings.page_fetch_timeout),
            ) as response:
                if response.status != 200:
                    return ""
                return (await response.text(errors="ignore"))[: self.settings.page_cleaned_max_chars]
        except Exception:
            return ""

    async def _extract_readability(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            async with session.get(
                url,
                headers={"User-Agent": "deep-research-mcp/1.0"},
                timeout=aiohttp.ClientTimeout(total=self.settings.page_fetch_timeout),
            ) as response:
                if response.status != 200:
                    return ""
                raw = await response.text(errors="ignore")
        except Exception:
            return ""
        return self.clean_html(raw, title="")

    async def _extract_pdf(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            import fitz  # type: ignore
        except Exception:
            logger.debug("PDF extraction skipped because pymupdf is unavailable.")
            return ""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.settings.page_fetch_timeout)) as response:
                if response.status != 200:
                    return ""
                payload = await response.read()
            with fitz.open(stream=payload, filetype="pdf") as pdf:
                text = "\n\n".join(page.get_text("text") for page in pdf)
            return re.sub(r"\n{3,}", "\n\n", text).strip()[: self.settings.page_cleaned_max_chars]
        except Exception:
            return ""

    @staticmethod
    def clean_html(raw_content: str, *, title: str = "") -> str:
        value = str(raw_content or "").strip()
        if not value:
            return ""
        probe = value[:600].lower()
        if "<html" not in probe and "<body" not in probe and "<div" not in probe:
            return value
        try:
            main_html = ReadabilityDocument(value).summary(html_partial=True)
            root = lxml_html.fragment_fromstring(main_html, create_parent="div")
            parts: List[str] = [f"# {title}"] if title else []
            for node in root.xpath(".//h1|.//h2|.//h3|.//p|.//li|.//blockquote|.//pre|.//table"):
                text = " ".join(node.itertext())
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    parts.append(text)
            return "\n\n".join(parts).strip()
        except Exception:
            return value

    def normalize_document(self, item: Dict[str, Any], *, source_name: str, source_layer: str) -> Document:
        url = str(item.get("url") or item.get("href") or item.get("link") or "").strip()
        pdf_url = str(item.get("pdf_url") or "").strip()
        title = str(item.get("title") or item.get("name") or "").strip()
        raw_content = str(item.get("raw_content") or item.get("content") or item.get("body") or item.get("snippet") or item.get("abstract") or "").strip()
        content = self.clean_html(raw_content, title=title)[: self.settings.page_cleaned_max_chars]
        source_kind = str(item.get("source_kind") or "").strip() or self._infer_source_kind(url or pdf_url, title, content)
        year = item.get("year")
        try:
            parsed_year = int(year) if year not in (None, "") else None
        except Exception:
            parsed_year = None
        doc: Document = {
            "document_id": self._document_id(url or pdf_url or title),
            "url": url or pdf_url,
            "title": title or url or pdf_url,
            "content": content,
            "raw_content": raw_content,
            "source_name": source_name,
            "source_layer": source_layer,
            "source_kind": source_kind,
            "page_type": source_kind,
            "source_type": "primary_source" if source_kind in {"paper", "pdf", "repo"} or source_layer == "vertical" else "analysis",
            "published_time": str(item.get("published_time") or item.get("published_date") or item.get("date") or ""),
            "authors": [str(a).strip() for a in item.get("authors", []) if str(a).strip()] if isinstance(item.get("authors"), list) else [],
            "venue": str(item.get("venue") or ""),
            "year": parsed_year,
            "doi": str(item.get("doi") or ""),
            "pdf_url": pdf_url,
            "metadata": dict(item.get("metadata") or {}),
            "score": float(item.get("score", item.get("provider_score", 0.0)) or 0.0),
        }
        return doc

    @staticmethod
    def _document_id(value: str) -> str:
        digest = hashlib.sha1(str(value or "document").encode("utf-8")).hexdigest()
        return f"DOC-{digest[:12]}"

    @staticmethod
    def _infer_source_kind(url: str, title: str, content: str) -> str:
        domain = urlparse(url).netloc.lower()
        path = urlparse(url).path.lower()
        lowered = f"{title} {content[:800]}".lower()
        if path.endswith(".pdf"):
            return "pdf"
        if "arxiv.org" in domain or "semanticscholar.org" in domain or "doi.org" in domain:
            return "paper"
        if "github.com" in domain:
            return "repo"
        if any(token in lowered for token in ("abstract", "citation", "journal", "conference")):
            return "paper"
        if any(token in path for token in ("/news", "/release", "/announcement")):
            return "news"
        return "web"

    def dedupe_and_rank(self, documents: List[Document]) -> List[Document]:
        by_key: Dict[str, Document] = {}
        for doc in documents:
            url = str(doc.get("url") or "").split("#", 1)[0].rstrip("/")
            if not url:
                continue
            candidate = dict(doc)
            candidate["content"] = str(candidate.get("content") or candidate.get("raw_content") or "").strip()
            if not candidate["content"]:
                continue
            key = url.lower()
            existing = by_key.get(key)
            if existing is None or self._rank_score(candidate) > self._rank_score(existing):
                by_key[key] = candidate
        return sorted(by_key.values(), key=self._rank_score, reverse=True)

    @staticmethod
    def _rank_score(document: Document) -> float:
        content_len = min(1.0, len(str(document.get("content") or "")) / 3000.0)
        score = float(document.get("score") or 0.0) + content_len
        if document.get("source_layer") == "vertical":
            score += 0.25
        if document.get("source_kind") in {"paper", "pdf", "repo"}:
            score += 0.15
        return score


class TavilyRetriever:
    name = "tavily"
    source_layer = "general"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        if not self.settings.tavily_api_key:
            return []
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "include_raw_content": bool(profile.get("include_raw_content", True)),
            "max_results": int(profile.get("max_results") or self.settings.search_max_results),
            "topic": str(profile.get("topic") or "general"),
            "search_depth": str(profile.get("search_depth") or "basic"),
        }
        exclude = [str(item).strip() for item in profile.get("exclude_domains", []) if str(item).strip()]
        if exclude:
            payload["exclude_domains"] = exclude[:10]
        async with session.post("https://api.tavily.com/search", json=payload) as response:
            if response.status != 200:
                return []
            data = await response.json()
        return [
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "raw_content": item.get("raw_content") or item.get("content", ""),
                "published_time": item.get("published_date") or "",
                "score": float(item.get("score", 0.0) or 0.0),
            }
            for item in data.get("results", [])
        ]


class DuckDuckGoRetriever:
    name = "duckduckgo"
    source_layer = "general"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        limit = int(profile.get("max_results") or self.settings.search_max_results)
        if DDGS is None:
            return []
        try:
            raw_results = await asyncio.to_thread(DDGS(timeout=10).text, query, max_results=limit)
        except Exception:
            return []
        return [
            {
                "url": item.get("href", ""),
                "title": item.get("title", ""),
                "raw_content": item.get("body", ""),
                "score": 0.0,
            }
            for item in raw_results[:limit]
        ]


class ExaRetriever:
    name = "exa"
    source_layer = "general"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        if not self.settings.exa_api_key:
            return []
        payload = {"query": query, "numResults": min(int(profile.get("max_results") or 5), 5), "type": "neural", "contents": {"text": {"maxCharacters": 3000}}}
        headers = {"x-api-key": self.settings.exa_api_key, "Content-Type": "application/json"}
        async with session.post("https://api.exa.ai/search", json=payload, headers=headers) as response:
            if response.status != 200:
                return []
            data = await response.json()
        return [
            {"url": r.get("url", ""), "title": r.get("title", ""), "raw_content": r.get("text", ""), "published_time": r.get("publishedDate", ""), "score": float(r.get("score", 0.0) or 0.0)}
            for r in data.get("results", [])
        ]


class SerperRetriever:
    name = "serper"
    source_layer = "general"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        if not self.settings.serper_api_key:
            return []
        payload = {"q": query, "num": min(int(profile.get("max_results") or 5), 5)}
        headers = {"X-API-KEY": self.settings.serper_api_key, "Content-Type": "application/json"}
        async with session.post("https://google.serper.dev/search", json=payload, headers=headers) as response:
            if response.status != 200:
                return []
            data = await response.json()
        return [
            {"url": r.get("link", ""), "title": r.get("title", ""), "raw_content": r.get("snippet", ""), "published_time": r.get("date", ""), "score": max(0.0, 10.0 - float(r.get("position", 10) or 10))}
            for r in data.get("organic", [])
        ]


class BochaRetriever:
    name = "bocha"
    source_layer = "general"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        if not self.settings.bocha_api_key:
            return []
        payload = {"query": query, "count": min(int(profile.get("max_results") or 5), 5), "freshness": "noLimit"}
        headers = {"Authorization": f"Bearer {self.settings.bocha_api_key}", "Content-Type": "application/json"}
        async with session.post("https://api.bochaai.com/v1/web-search", json=payload, headers=headers) as response:
            if response.status != 200:
                return []
            data = await response.json()
        pages = data.get("data", {}).get("webPages", {}).get("value", [])
        return [
            {"url": p.get("url", ""), "title": p.get("name", ""), "raw_content": p.get("snippet", ""), "published_time": p.get("dateLastCrawled", ""), "score": 0.5}
            for p in pages
        ]


class SerpApiRetriever:
    name = "serpapi"
    source_layer = "general"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        if not self.settings.serpapi_api_key:
            return []
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.settings.serpapi_api_key,
            "num": min(int(profile.get("max_results") or 5), 10),
        }
        async with session.get("https://serpapi.com/search.json", params=params) as response:
            if response.status != 200:
                return []
            data = await response.json()
        return [
            {
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "raw_content": item.get("snippet", ""),
                "published_time": item.get("date", ""),
                "score": max(0.0, 10.0 - float(item.get("position", 10) or 10)),
            }
            for item in data.get("organic_results", [])
        ]


class BingRetriever:
    name = "bing"
    source_layer = "general"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        if not self.settings.bing_api_key:
            return []
        headers = {"Ocp-Apim-Subscription-Key": self.settings.bing_api_key}
        params = {"q": query, "count": min(int(profile.get("max_results") or 5), 10), "textDecorations": "false", "textFormat": "Raw"}
        async with session.get("https://api.bing.microsoft.com/v7.0/search", params=params, headers=headers) as response:
            if response.status != 200:
                return []
            data = await response.json()
        return [
            {
                "url": item.get("url", ""),
                "title": item.get("name", ""),
                "raw_content": item.get("snippet", ""),
                "published_time": item.get("dateLastCrawled", ""),
                "score": 0.5,
            }
            for item in data.get("webPages", {}).get("value", [])
        ]


class GoogleRetriever:
    name = "google"
    source_layer = "general"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        if not (self.settings.google_api_key and self.settings.google_cse_id):
            return []
        params = {
            "key": self.settings.google_api_key,
            "cx": self.settings.google_cse_id,
            "q": query,
            "num": min(int(profile.get("max_results") or 5), 10),
        }
        async with session.get("https://www.googleapis.com/customsearch/v1", params=params) as response:
            if response.status != 200:
                return []
            data = await response.json()
        return [
            {
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "raw_content": item.get("snippet", ""),
                "published_time": "",
                "score": 0.5,
            }
            for item in data.get("items", [])
        ]


class SearxRetriever:
    name = "searx"
    source_layer = "general"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        base = str(self.settings.searx_base_url or "").rstrip("/")
        if not base:
            return []
        params = {"q": query, "format": "json", "language": "auto"}
        async with session.get(f"{base}/search", params=params) as response:
            if response.status != 200:
                return []
            data = await response.json()
        return [
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "raw_content": item.get("content", ""),
                "published_time": item.get("publishedDate", ""),
                "score": float(item.get("score", 0.0) or 0.0),
            }
            for item in data.get("results", [])[: int(profile.get("max_results") or 5)]
        ]


class SemanticScholarRetriever:
    name = "semantic_scholar"
    source_layer = "vertical"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        params = {"query": query, "limit": min(int(profile.get("max_results") or 3), 5), "fields": "title,abstract,url,year,citationCount,authors,venue,externalIds,openAccessPdf"}
        try:
            async with session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    return []
                data = await response.json()
        except Exception:
            return []
        docs = []
        for paper in data.get("data", []):
            external = paper.get("externalIds") if isinstance(paper.get("externalIds"), dict) else {}
            pdf = paper.get("openAccessPdf") if isinstance(paper.get("openAccessPdf"), dict) else {}
            docs.append(
                {
                    "url": paper.get("url", ""),
                    "title": paper.get("title", ""),
                    "raw_content": paper.get("abstract") or "",
                    "year": paper.get("year"),
                    "authors": [a.get("name", "") for a in paper.get("authors", []) if isinstance(a, dict)],
                    "venue": paper.get("venue", ""),
                    "doi": external.get("DOI", ""),
                    "pdf_url": pdf.get("url", ""),
                    "source_kind": "paper",
                    "score": min(float(paper.get("citationCount", 0) or 0) / 100.0, 1.0),
                }
            )
        return docs


class ArxivRetriever:
    name = "arxiv"
    source_layer = "vertical"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(int(profile.get("max_results") or 5), 10),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            async with session.get("https://export.arxiv.org/api/query", params=params) as response:
                if response.status != 200:
                    return []
                payload = await response.text()
        except Exception:
            return []
        try:
            root = ET.fromstring(payload)
        except Exception:
            return []
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        docs = []
        for entry in root.findall("atom:entry", ns):
            title = _xml_text(entry.find("atom:title", ns))
            summary = _xml_text(entry.find("atom:summary", ns))
            url = _xml_text(entry.find("atom:id", ns))
            published = _xml_text(entry.find("atom:published", ns))
            authors = [_xml_text(author.find("atom:name", ns)) for author in entry.findall("atom:author", ns)]
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            docs.append({
                "url": url,
                "title": title,
                "raw_content": summary,
                "published_time": published,
                "authors": [a for a in authors if a],
                "venue": "arXiv",
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "pdf_url": pdf_url,
                "source_kind": "paper",
                "score": 0.75,
            })
        return docs


class PubMedCentralRetriever:
    name = "pubmed_central"
    source_layer = "vertical"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, session, query, profile):
        max_results = min(int(profile.get("max_results") or 5), 10)
        try:
            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pmc", "term": query, "retmode": "json", "retmax": max_results},
            ) as response:
                if response.status != 200:
                    return []
                search_data = await response.json()
            ids = search_data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pmc", "id": ",".join(ids), "retmode": "json"},
            ) as response:
                if response.status != 200:
                    return []
                summary_data = await response.json()
        except Exception:
            return []
        result = summary_data.get("result", {})
        docs = []
        for pmcid in ids:
            item = result.get(str(pmcid), {})
            if not isinstance(item, dict):
                continue
            title = item.get("title", "")
            authors = [a.get("name", "") for a in item.get("authors", []) if isinstance(a, dict)]
            pubdate = item.get("pubdate", "")
            docs.append({
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                "title": title,
                "raw_content": item.get("fulljournalname", ""),
                "published_time": pubdate,
                "authors": authors,
                "venue": item.get("fulljournalname", ""),
                "year": int(pubdate[:4]) if str(pubdate)[:4].isdigit() else None,
                "source_kind": "paper",
                "score": 0.65,
            })
        return docs


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _xml_text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return re.sub(r"\s+", " ", node.text).strip()
