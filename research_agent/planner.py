from __future__ import annotations

from anthropic import AsyncAnthropic

from .config import Settings
from .llm import AgentError, UsageTracker, base_kwargs, check_stop
from .models import ResearchPlan

SYSTEM = """You are the planner on a research team. Break the user's question into focused \
sub-questions that separate researchers can each investigate independently with web search.

Cover what the user needs in order to get a good answer, and skip trivia. Sub-questions should not \
overlap. Use at most {limit}; a narrow question may need only one or two."""


async def plan(
    client: AsyncAnthropic, settings: Settings, question: str, usage: UsageTracker
) -> ResearchPlan:
    response = await client.beta.messages.parse(
        **base_kwargs(settings),
        max_tokens=16000,
        system=SYSTEM.format(limit=settings.max_sub_questions),
        messages=[{"role": "user", "content": question}],
        output_format=ResearchPlan,
        output_config={"effort": settings.effort},
    )
    usage.add(response)
    check_stop("planner", response)
    result = response.parsed_output
    if result is None or not result.sub_questions:
        raise AgentError("planner: returned no sub-questions")
    result.sub_questions = result.sub_questions[: settings.max_sub_questions]
    return result
