import pytest

from research_agent import orchestrator
from research_agent.config import Settings
from research_agent.llm import AgentError
from research_agent.models import Critique, ResearchPlan, SubQuestion
from research_agent.sources import extract_finding

from .fakes import text


@pytest.fixture
def team(monkeypatch):
    """Replace every agent with a scripted fake and record what each was asked."""
    calls = {"research": [], "review": [], "write": []}
    state = {"follow_ups": [["follow-up?"], []], "fail": set()}

    async def plan(client, settings, question, usage):
        return ResearchPlan(sub_questions=[SubQuestion(question=q, rationale="r") for q in ("q1", "q2")])

    async def research(client, settings, main_question, sub_question, usage):
        calls["research"].append(sub_question)
        if sub_question in state["fail"]:
            raise AgentError("boom")
        return extract_finding(sub_question, [text(f"About {sub_question}.", (f"https://{sub_question}.example", sub_question, "x"))])

    async def review(client, settings, question, notes, usage, *, allow_follow_ups):
        calls["review"].append((notes, allow_follow_ups))
        follow_ups = state["follow_ups"].pop(0) if allow_follow_ups else []
        return Critique(confidence="medium", issues=[], gaps=[], follow_up_questions=follow_ups)

    async def write(client, settings, question, notes, critiques, usage, on_text=None):
        calls["write"].append(notes)
        return "# Report\n\nAnswer [1]."

    monkeypatch.setattr(orchestrator.planner, "plan", plan)
    monkeypatch.setattr(orchestrator.researcher, "research", research)
    monkeypatch.setattr(orchestrator.critic, "review", review)
    monkeypatch.setattr(orchestrator.writer, "write", write)
    return calls, state


async def test_runs_follow_up_round_then_writes(team):
    calls, _ = team
    result = await orchestrator.run_research("main?", Settings(max_rounds=3), client=object())

    assert calls["research"] == ["q1", "q2", "follow-up?"]
    assert len(calls["review"]) == 2
    assert [f.question for f in result.findings] == ["q1", "q2", "follow-up?"]
    assert result.report.startswith("# Report")
    assert "## Sources\n\n1. [q1](https://q1.example)" in result.report
    assert "3. [follow-up?](https://follow-up?.example)" in result.report


async def test_last_round_disallows_follow_ups(team):
    calls, _ = team
    await orchestrator.run_research("main?", Settings(max_rounds=1), client=object())
    assert calls["research"] == ["q1", "q2"]
    assert calls["review"][0][1] is False


async def test_failed_researcher_is_reported_not_fatal(team):
    calls, state = team
    state["fail"].add("q2")
    result = await orchestrator.run_research("main?", Settings(max_rounds=1), client=object())

    assert result.findings[1].error == "boom"
    assert "(Research failed: boom)" in calls["write"][0]


async def test_all_researchers_failing_aborts(team):
    _, state = team
    state["fail"].update({"q1", "q2"})
    with pytest.raises(AgentError):
        await orchestrator.run_research("main?", Settings(max_rounds=1), client=object())
