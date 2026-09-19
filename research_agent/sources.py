"""Turning researcher output into cited notes with one global source numbering."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import Finding, Segment, Source

MAX_QUOTES_PER_SOURCE = 3


def extract_finding(question: str, blocks: Iterable[Any]) -> Finding:
    """Build a Finding from a researcher's response content blocks.

    Text blocks carry web-search citations; fetched pages are kept as sources too,
    since web_fetch results arrive without per-sentence citations.
    """
    finding = Finding(question=question)
    for block in blocks:
        if block.type == "text":
            urls: list[str] = []
            for citation in block.citations or []:
                if citation.type != "web_search_result_location":
                    continue
                source = finding.sources.setdefault(
                    citation.url, Source(url=citation.url, title=citation.title or citation.url)
                )
                quote = (citation.cited_text or "").strip()
                if quote and quote not in source.quotes and len(source.quotes) < MAX_QUOTES_PER_SOURCE:
                    source.quotes.append(quote)
                if citation.url not in urls:
                    urls.append(citation.url)
            finding.segments.append(Segment(text=block.text, urls=urls))
        elif block.type == "web_fetch_tool_result" and block.content.type == "web_fetch_result":
            url = block.content.url
            title = block.content.content.title or url
            finding.sources.setdefault(url, Source(url=url, title=title))
    return finding


class SourceRegistry:
    """Assigns every distinct URL across all findings a stable number, in order of first use."""

    def __init__(self, findings: Iterable[Finding] = ()):
        self._numbers: dict[str, int] = {}
        self._sources: list[Source] = []
        for finding in findings:
            self.add(finding)

    def add(self, finding: Finding) -> None:
        for segment in finding.segments:
            for url in segment.urls:
                self._register(finding.sources[url])
        for source in finding.sources.values():
            self._register(source)

    def _register(self, source: Source) -> None:
        if source.url not in self._numbers:
            self._numbers[source.url] = len(self._sources) + 1
            self._sources.append(source)

    def number(self, url: str) -> int:
        return self._numbers[url]

    @property
    def sources(self) -> list[Source]:
        return list(self._sources)

    def render_finding(self, finding: Finding) -> str:
        if finding.error:
            return f"(Research failed: {finding.error})"
        parts = []
        for segment in finding.segments:
            markers = "".join(f"[{self.number(url)}]" for url in segment.urls)
            parts.append(segment.text + markers)
        return "".join(parts).strip() or "(No findings.)"

    def render_notes(self, findings: Iterable[Finding]) -> str:
        """All findings plus the numbered source list, as input for the critic and writer."""
        sections = [
            f"### {finding.question}\n\n{self.render_finding(finding)}" for finding in findings
        ]
        sections.append("### Sources\n\n" + self.render_source_list(with_quotes=True))
        return "\n\n".join(sections)

    def render_source_list(self, *, with_quotes: bool = False) -> str:
        if not self._sources:
            return "(none)"
        lines = []
        for index, source in enumerate(self._sources, start=1):
            lines.append(f"[{index}] {source.title} - {source.url}")
            if with_quotes:
                lines.extend(f'    > "{quote}"' for quote in source.quotes)
        return "\n".join(lines)
