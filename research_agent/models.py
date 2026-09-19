from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    question: str = Field(description="A focused question one researcher can answer with web search.")
    rationale: str = Field(description="One sentence on why answering it matters for the main question.")


class ResearchPlan(BaseModel):
    sub_questions: list[SubQuestion]


class Issue(BaseModel):
    claim: str = Field(description="The claim from the notes, quoted or closely paraphrased.")
    problem: str = Field(description="What is wrong with it: unsupported, contradicted, outdated, weak source, etc.")
    severity: Literal["low", "medium", "high"]


class Critique(BaseModel):
    confidence: Literal["low", "medium", "high"] = Field(
        description="How well the notes, as they stand, support a good answer to the question."
    )
    issues: list[Issue]
    gaps: list[str] = Field(description="Parts of the question the notes leave unanswered.")
    follow_up_questions: list[str] = Field(
        description="New sub-questions worth another round of web research. Empty if none are needed."
    )


@dataclass
class Source:
    url: str
    title: str
    quotes: list[str] = field(default_factory=list)


@dataclass
class Segment:
    """A run of researcher text plus the URLs of the sources it cites."""

    text: str
    urls: list[str] = field(default_factory=list)


@dataclass
class Finding:
    question: str
    segments: list[Segment] = field(default_factory=list)
    sources: dict[str, Source] = field(default_factory=dict)
    error: str | None = None
