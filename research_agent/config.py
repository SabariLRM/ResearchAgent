from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class Settings:
    """Knobs for a research run. Defaults favour quality over cost."""

    model: str = "claude-opus-5"
    # Effort for the planner, critic and writer, where judgement matters most.
    effort: Effort = "high"
    # Researchers mostly search and summarise, so they run cheaper.
    research_effort: Effort = "medium"
    max_sub_questions: int = 5
    # Round 1 researches the plan; each extra round researches the critic's follow-ups.
    max_rounds: int = 2
    max_follow_ups: int = 3
    concurrency: int = 4
    max_searches: int = 6
    max_fetches: int = 4
    # Server-side refusal fallbacks: a declined request is retried on Anthropic's
    # recommended fallback model instead of failing.
    use_fallbacks: bool = True
