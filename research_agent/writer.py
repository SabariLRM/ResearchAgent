from __future__ import annotations

from collections.abc import Callable

from anthropic import AsyncAnthropic

from .config import Settings
from .llm import AgentError, UsageTracker, base_kwargs, check_stop
from .models import Critique

SYSTEM = """You are the writer on a research team. Write the final report answering the user's \
question in Markdown, using only the researchers' notes and the critic's reviews.

- Open with a title, then a direct answer in two to four sentences. Follow with the sections the \
question actually needs.
- Cite sources inline by number, like [3] or [2][5]. Use only numbers from the source list.
- Where the critic flagged a claim, drop it or state the uncertainty plainly.
- Match the length to the question: cover the substance without filler sections, repeated \
summaries, or boilerplate.
- Do not add a sources or references section; it is appended automatically."""


def format_critiques(critiques: list[Critique]) -> str:
    parts = []
    for round_no, critique in enumerate(critiques, start=1):
        lines = [f"Round {round_no} - confidence: {critique.confidence}"]
        lines += [f"- [{i.severity}] {i.claim}: {i.problem}" for i in critique.issues]
        lines += [f"- Gap: {gap}" for gap in critique.gaps]
        parts.append("\n".join(lines))
    return "\n\n".join(parts) or "(no review)"


async def write(
    client: AsyncAnthropic,
    settings: Settings,
    question: str,
    notes: str,
    critiques: list[Critique],
    usage: UsageTracker,
    on_text: Callable[[str], None] | None = None,
) -> str:
    content = (
        f"<question>\n{question}\n</question>\n\n"
        f"<notes>\n{notes}\n</notes>\n\n"
        f"<critic_reviews>\n{format_critiques(critiques)}\n</critic_reviews>"
    )
    async with client.beta.messages.stream(
        **base_kwargs(settings),
        max_tokens=64000,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": settings.effort},
    ) as stream:
        if on_text:
            async for text in stream.text_stream:
                on_text(text)
        response = await stream.get_final_message()
    usage.add(response)
    check_stop("writer", response)
    report = "".join(block.text for block in response.content if block.type == "text").strip()
    if not report:
        raise AgentError("writer: produced an empty report")
    return report
