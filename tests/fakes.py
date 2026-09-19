"""Minimal stand-ins for the Anthropic client so tests run offline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from anthropic.types.beta import (
    BetaCitationsWebSearchResultLocation,
    BetaMessage,
    BetaTextBlock,
    BetaUsage,
)


def text(content: str, *cites: tuple[str, str, str]) -> BetaTextBlock:
    """A text block citing (url, title, quote) triples."""
    return BetaTextBlock(
        type="text",
        text=content,
        citations=[
            BetaCitationsWebSearchResultLocation(
                type="web_search_result_location", url=url, title=title, cited_text=quote, encrypted_index="x"
            )
            for url, title, quote in cites
        ]
        or None,
    )


def message(content: list[Any], stop_reason: str = "end_turn") -> BetaMessage:
    return BetaMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=content,
        stop_reason=stop_reason,
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )


class _Stream:
    def __init__(self, response: BetaMessage):
        self._response = response

    async def __aenter__(self) -> "_Stream":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get_final_message(self) -> BetaMessage:
        return self._response


class FakeStreamingClient:
    """Returns queued responses from beta.messages.stream and records each request."""

    def __init__(self, responses: list[BetaMessage]):
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs: Any) -> _Stream:
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        return _Stream(self.responses.pop(0))
