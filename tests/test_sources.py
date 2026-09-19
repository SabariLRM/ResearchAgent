from types import SimpleNamespace

from research_agent.sources import SourceRegistry, extract_finding

from .fakes import text

A = ("https://a.example", "Source A", "quote a")
B = ("https://b.example", "Source B", "quote b")


def test_extract_finding_collects_cited_segments_and_sources():
    blocks = [
        SimpleNamespace(type="server_tool_use"),
        text("Intro. "),
        text("Claim one.", A),
        text(" Claim two.", A, B),
    ]
    finding = extract_finding("q", blocks)

    assert [s.urls for s in finding.segments] == [[], [A[0]], [A[0], B[0]]]
    assert list(finding.sources) == [A[0], B[0]]
    assert finding.sources[A[0]].quotes == ["quote a"]


def test_extract_finding_keeps_fetched_pages_as_sources():
    fetch = SimpleNamespace(
        type="web_fetch_tool_result",
        content=SimpleNamespace(
            type="web_fetch_result", url="https://f.example", content=SimpleNamespace(title="Fetched")
        ),
    )
    finding = extract_finding("q", [fetch, text("Notes.")])
    assert finding.sources["https://f.example"].title == "Fetched"


def test_registry_numbers_sources_globally_and_renders_markers():
    first = extract_finding("q1", [text("Alpha.", A)])
    second = extract_finding("q2", [text("Beta.", B, A)])
    registry = SourceRegistry([first, second])

    assert registry.render_finding(first) == "Alpha.[1]"
    assert registry.render_finding(second) == "Beta.[2][1]"
    notes = registry.render_notes([first, second])
    assert "### q2" in notes
    assert "[2] Source B - https://b.example" in notes
    assert '> "quote b"' in notes
