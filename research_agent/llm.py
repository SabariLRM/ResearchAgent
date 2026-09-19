"""Pieces shared by every agent: request options, stop-reason checks, usage tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anthropic.types.beta import BetaMessage

from .config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AgentError(RuntimeError):
    """An agent could not produce a usable response."""


class RefusalError(AgentError):
    def __init__(self, agent: str, message: BetaMessage):
        details = message.stop_details
        self.category = details.category if details else None
        super().__init__(f"{agent}: request was declined (category: {self.category or 'unspecified'})")


def base_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": settings.model}
    if settings.use_fallbacks:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs


def check_stop(agent: str, message: BetaMessage, *, allow_truncation: bool = False) -> None:
    if message.stop_reason == "refusal":
        raise RefusalError(agent, message)
    if message.stop_reason == "max_tokens" and not allow_truncation:
        raise AgentError(f"{agent}: response hit max_tokens and was cut off")


@dataclass
class UsageTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    web_searches: int = 0
    web_fetches: int = 0
    requests: int = 0

    def add(self, message: BetaMessage) -> None:
        usage = message.usage
        self.requests += 1
        self.input_tokens += usage.input_tokens + (usage.cache_creation_input_tokens or 0)
        self.cache_read_tokens += usage.cache_read_input_tokens or 0
        self.output_tokens += usage.output_tokens
        if usage.server_tool_use:
            self.web_searches += usage.server_tool_use.web_search_requests
            self.web_fetches += usage.server_tool_use.web_fetch_requests

    def summary(self) -> str:
        return (
            f"{self.requests} requests · {self.input_tokens:,} input tokens "
            f"(+{self.cache_read_tokens:,} cached) · {self.output_tokens:,} output tokens · "
            f"{self.web_searches} searches · {self.web_fetches} page fetches"
        )
