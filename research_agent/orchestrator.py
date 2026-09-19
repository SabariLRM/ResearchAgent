"""Runs the team: plan -> parallel research -> critic (-> follow-up research) -> writer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import anthropic
from anthropic import AsyncAnthropic

from . import critic, planner, researcher, writer
from .config import Settings
from .llm import AgentError, UsageTracker
from .models import Critique, Finding, ResearchPlan
from .sources import SourceRegistry


@dataclass
class ResearchResult:
    question: str
    plan: ResearchPlan
    findings: list[Finding]
    critiques: list[Critique]
    report: str
    usage: UsageTracker = field(default_factory=UsageTracker)


@dataclass
class Callbacks:
    on_status: Callable[[str], None] = lambda message: None
    on_plan: Callable[[ResearchPlan], None] = lambda plan: None
    on_finding: Callable[[Finding], None] = lambda finding: None
    on_critique: Callable[[Critique], None] = lambda critique: None
    on_report_text: Callable[[str], None] | None = None


async def run_research(
    question: str,
    settings: Settings | None = None,
    *,
    client: AsyncAnthropic | None = None,
    callbacks: Callbacks | None = None,
) -> ResearchResult:
    settings = settings or Settings()
    client = client or AsyncAnthropic()
    cb = callbacks or Callbacks()
    usage = UsageTracker()

    cb.on_status("Planning")
    research_plan = await planner.plan(client, settings, question, usage)
    cb.on_plan(research_plan)

    semaphore = asyncio.Semaphore(settings.concurrency)

    async def investigate(sub_question: str) -> Finding:
        async with semaphore:
            try:
                finding = await researcher.research(client, settings, question, sub_question, usage)
            except (AgentError, anthropic.APIError) as exc:
                # One failed researcher shouldn't sink the run; the critic sees the gap.
                finding = Finding(question=sub_question, error=str(exc))
        cb.on_finding(finding)
        return finding

    findings: list[Finding] = []
    critiques: list[Critique] = []
    pending = [sub.question for sub in research_plan.sub_questions]

    for round_no in range(1, settings.max_rounds + 1):
        cb.on_status(f"Research round {round_no}: {len(pending)} researcher(s)")
        findings += await asyncio.gather(*(investigate(q) for q in pending))
        if all(f.error for f in findings):
            raise AgentError("every researcher failed; see the errors above")

        cb.on_status("Critic reviewing the notes")
        notes = SourceRegistry(findings).render_notes(findings)
        is_last_round = round_no == settings.max_rounds
        critique = await critic.review(
            client, settings, question, notes, usage, allow_follow_ups=not is_last_round
        )
        critiques.append(critique)
        cb.on_critique(critique)
        pending = critique.follow_up_questions
        if not pending:
            break

    cb.on_status("Writing the report")
    registry = SourceRegistry(findings)
    body = await writer.write(
        client,
        settings,
        question,
        registry.render_notes(findings),
        critiques,
        usage,
        on_text=cb.on_report_text,
    )
    report = f"{body}\n\n## Sources\n\n" + "\n".join(
        f"{i}. [{s.title}]({s.url})" for i, s in enumerate(registry.sources, start=1)
    )
    return ResearchResult(question, research_plan, findings, critiques, report + "\n", usage)
