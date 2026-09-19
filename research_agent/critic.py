from __future__ import annotations

from anthropic import AsyncAnthropic

from .config import Settings
from .llm import AgentError, UsageTracker, base_kwargs, check_stop
from .models import Critique

SYSTEM = """You are the critic on a research team. You receive the user's question and the \
researchers' notes, with numbered sources and the passages quoted from them. Judge whether the \
notes support a good answer.

- Report every problem claim you find: unsupported by its cited source, contradicted by another \
note, likely outdated, or resting on a weak source. Rate each by severity.
- List gaps: parts of the question the notes leave unanswered.
- Propose follow-up questions only for gaps worth another round of web research, at most {limit}. \
Return an empty list when the notes are good enough to write the answer."""


async def review(
    client: AsyncAnthropic,
    settings: Settings,
    question: str,
    notes: str,
    usage: UsageTracker,
    *,
    allow_follow_ups: bool,
) -> Critique:
    limit = settings.max_follow_ups if allow_follow_ups else 0
    response = await client.beta.messages.parse(
        **base_kwargs(settings),
        max_tokens=16000,
        system=SYSTEM.format(limit=limit),
        messages=[
            {
                "role": "user",
                "content": f"<question>\n{question}\n</question>\n\n<notes>\n{notes}\n</notes>",
            }
        ],
        output_format=Critique,
        output_config={"effort": settings.effort},
    )
    usage.add(response)
    check_stop("critic", response)
    critique = response.parsed_output
    if critique is None:
        raise AgentError("critic: returned no review")
    critique.follow_up_questions = critique.follow_up_questions[:limit]
    return critique
