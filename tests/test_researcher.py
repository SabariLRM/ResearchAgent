import pytest
from anthropic.types.beta import BetaServerToolUseBlock

from research_agent.config import Settings
from research_agent.llm import FALLBACK_BETA, RefusalError, UsageTracker
from research_agent.researcher import research

from .fakes import FakeStreamingClient, message, text

URL = ("https://a.example", "A", "q")


async def test_resumes_paused_turn_and_merges_content():
    paused_tool = BetaServerToolUseBlock(type="server_tool_use", id="srv_1", name="web_search", input={"query": "x"})
    client = FakeStreamingClient(
        [
            message([text("Part one.", URL), paused_tool], stop_reason="pause_turn"),
            message([text(" Part two.")]),
        ]
    )
    usage = UsageTracker()

    finding = await research(client, Settings(), "main?", "sub?", usage)

    assert "".join(s.text for s in finding.segments) == "Part one. Part two."
    assert usage.requests == 2
    resumed = client.requests[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]
    assert resumed[1]["content"][-1].type == "server_tool_use"


async def test_sends_fallbacks_and_web_tools():
    client = FakeStreamingClient([message([text("Done.")])])
    await research(client, Settings(max_searches=3), "main?", "sub?", UsageTracker())

    request = client.requests[0]
    assert request["fallbacks"] == "default"
    assert request["betas"] == [FALLBACK_BETA]
    assert {t["type"] for t in request["tools"]} == {"web_search_20260209", "web_fetch_20260209"}
    assert request["tools"][0]["max_uses"] == 3


async def test_no_fallbacks_when_disabled():
    client = FakeStreamingClient([message([text("Done.")])])
    await research(client, Settings(use_fallbacks=False), "main?", "sub?", UsageTracker())
    assert "fallbacks" not in client.requests[0]


async def test_refusal_raises():
    client = FakeStreamingClient([message([], stop_reason="refusal")])
    with pytest.raises(RefusalError):
        await research(client, Settings(), "main?", "sub?", UsageTracker())
