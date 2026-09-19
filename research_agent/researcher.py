from __future__ import annotations

import datetime as dt
from typing import Any

from anthropic import AsyncAnthropic

from .config import Settings
from .llm import AgentError, UsageTracker, base_kwargs, check_stop
from .models import Finding
from .sources import extract_finding

# Server-side search loops pause after a fixed number of iterations; resume at most this many times.
MAX_CONTINUATIONS = 4

SYSTEM = """You are a researcher on a research team. Investigate one sub-question using web search \
and web fetch, then write up what you found for the rest of the team.

Today's date is {today}.

- Prefer primary and authoritative sources such as official documentation, papers, and reputable \
outlets. Say so when sources disagree or when information may be out of date.
- Back factual claims with search results so they carry citations.
- Your write-up is notes for a colleague, not a finished report: key findings in short paragraphs \
or bullets, then one line on what remains uncertain. No preamble."""


async def research(
    client: AsyncAnthropic,
    settings: Settings,
    main_question: str,
    sub_question: str,
    usage: UsageTracker,
) -> Finding:
    tools: list[dict[str, Any]] = [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": settings.max_searches},
        {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": settings.max_fetches},
    ]
    prompt = f"Overall research question: {main_question}\n\nYour sub-question: {sub_question}"
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    blocks: list[Any] = []

    for _ in range(MAX_CONTINUATIONS + 1):
        async with client.beta.messages.stream(
            **base_kwargs(settings),
            max_tokens=64000,
            system=SYSTEM.format(today=dt.date.today().isoformat()),
            tools=tools,
            messages=messages,
            output_config={"effort": settings.research_effort},
        ) as stream:
            response = await stream.get_final_message()
        usage.add(response)
        check_stop("researcher", response, allow_truncation=True)
        blocks.extend(response.content)
        if response.stop_reason != "pause_turn":
            break
        # Resend the paused turn; the server picks up where it left off.
        messages = [messages[0], {"role": "assistant", "content": list(blocks)}]
    else:
        raise AgentError("researcher: search loop kept pausing; gave up")

    finding = extract_finding(sub_question, blocks)
    if not finding.segments:
        raise AgentError("researcher: produced no notes")
    return finding
