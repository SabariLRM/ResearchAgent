from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import re
import sys
from pathlib import Path

import anthropic
from rich.console import Console

from .config import Settings
from .llm import AgentError
from .models import Critique, Finding, ResearchPlan
from .orchestrator import Callbacks, run_research

console = Console(stderr=True)


def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "report"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = Settings()
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Answer a question with a team of Claude agents: planner, web researchers, critic, writer.",
    )
    parser.add_argument("question", nargs="+", help="the question to research")
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--effort", default=defaults.effort, choices=["low", "medium", "high", "xhigh", "max"],
                        help="effort for planner, critic and writer (default: %(default)s)")
    parser.add_argument("--research-effort", default=defaults.research_effort,
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="effort for researchers (default: %(default)s)")
    parser.add_argument("--sub-questions", type=int, default=defaults.max_sub_questions,
                        help="max sub-questions in the plan (default: %(default)s)")
    parser.add_argument("--rounds", type=int, default=defaults.max_rounds,
                        help="max research rounds, including critic follow-ups (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency,
                        help="researchers running at once (default: %(default)s)")
    parser.add_argument("--max-searches", type=int, default=defaults.max_searches,
                        help="web searches per researcher (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="output directory (default: %(default)s)")
    parser.add_argument("--no-fallbacks", action="store_true",
                        help="disable server-side refusal fallbacks to another model")
    parser.add_argument("--quiet", action="store_true", help="don't stream the report to the terminal")
    return parser.parse_args(argv)


def build_callbacks(quiet: bool) -> Callbacks:
    def on_plan(plan: ResearchPlan) -> None:
        console.print("[bold]Plan[/bold]")
        for i, sub in enumerate(plan.sub_questions, start=1):
            console.print(f"  {i}. {sub.question} [dim]- {sub.rationale}[/dim]")

    def on_finding(finding: Finding) -> None:
        if finding.error:
            console.print(f"  [red]✗[/red] {finding.question} [dim]({finding.error})[/dim]")
        else:
            console.print(f"  [green]✓[/green] {finding.question} [dim]({len(finding.sources)} sources)[/dim]")

    def on_critique(critique: Critique) -> None:
        console.print(
            f"  confidence: {critique.confidence} · {len(critique.issues)} issue(s) · {len(critique.gaps)} gap(s)"
        )
        for follow_up in critique.follow_up_questions:
            console.print(f"  [yellow]→[/yellow] follow-up: {follow_up}")

    return Callbacks(
        on_status=lambda message: console.print(f"\n[bold cyan]▶ {message}[/bold cyan]"),
        on_plan=on_plan,
        on_finding=on_finding,
        on_critique=on_critique,
        on_report_text=None if quiet else lambda text: print(text, end="", flush=True),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    question = " ".join(args.question)
    settings = Settings(
        model=args.model,
        effort=args.effort,
        research_effort=args.research_effort,
        max_sub_questions=args.sub_questions,
        max_rounds=args.rounds,
        concurrency=args.concurrency,
        max_searches=args.max_searches,
        use_fallbacks=not args.no_fallbacks,
    )

    try:
        result = asyncio.run(run_research(question, settings, callbacks=build_callbacks(args.quiet)))
    except anthropic.AuthenticationError:
        console.print("[red]Authentication failed.[/red] Set ANTHROPIC_API_KEY or run `ant auth login`.")
        sys.exit(1)
    except (AgentError, anthropic.APIError) as exc:
        console.print(f"[red]Research failed:[/red] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{dt.date.today().isoformat()}-{slugify(question)}.md"
    path.write_text(result.report, encoding="utf-8")
    print()
    console.print(f"\n[bold green]Saved[/bold green] {path}")
    console.print(f"[dim]{result.usage.summary()}[/dim]")
