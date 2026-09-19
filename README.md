# ResearchAgent

A multi-agent research assistant built on Claude. You ask a question, and a small team of agents turns it into a short, sourced Markdown report.

```
                   ┌──────────────┐
  question ──────▶ │   Planner    │  splits the question into focused sub-questions
                   └──────┬───────┘
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ Researcher │  │ Researcher │  │ Researcher │   run in parallel, each with
   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘   web search + web fetch
          └───────────────┼───────────────┘
                          ▼
                   ┌──────────────┐   flags weak or unsupported claims and gaps;
                   │    Critic    │── can send follow-up questions back to new
                   └──────┬───────┘   researchers for another round
                          ▼
                   ┌──────────────┐
                   │    Writer    │  writes the final report with numbered citations
                   └──────────────┘
```

## How it works

| Agent | What it does | Claude feature used |
|---|---|---|
| **Planner** | Breaks the question into up to 5 independent sub-questions. | Structured outputs (Pydantic schema) |
| **Researchers** | One per sub-question, running concurrently. Each searches the web, reads pages and writes cited notes. | Server-side `web_search` / `web_fetch` tools, streaming, `pause_turn` resumption |
| **Critic** | Reviews all notes against their sources. Flags unsupported, contradicted or outdated claims, lists gaps, and proposes follow-up questions. | Structured outputs |
| **Writer** | Writes the final Markdown report, citing sources by number and hedging anything the critic flagged. | Streaming |

The orchestrator (`research_agent/orchestrator.py`) is plain `asyncio` code. The critic can trigger more research rounds (`--rounds`, default 2), and a failed researcher doesn't stop the run: the failure is passed to the critic and writer as a gap.

Citations come straight from the web search results. Every URL gets one global number across all researchers, so `[3]` means the same source everywhere, and the report ends with a matching **Sources** list.

## Setup

Requires Python 3.10+ and an Anthropic API key.

```bash
git clone https://github.com/SabariLRM/ResearchAgent.git
cd ResearchAgent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

export ANTHROPIC_API_KEY=sk-ant-...
```

Web search has to be enabled for your organization in the [Claude Console](https://platform.claude.com/) (it's under privacy/feature settings).

## Usage

```bash
research-agent "What's the best budget NAS setup for a Raspberry Pi 5?"
```

You'll see the plan, each researcher finishing, the critic's verdict, and then the report streaming in. The report is saved to `reports/YYYY-MM-DD-<slug>.md`.

Useful options:

```bash
research-agent "question" \
  --sub-questions 3        # smaller plan, cheaper run
  --rounds 1               # skip the critic's follow-up research round
  --research-effort low    # faster, cheaper researchers
  --effort xhigh           # more careful planner, critic and writer
  --concurrency 2          # fewer researchers at once (helps with rate limits)
  --out ~/notes/research   # where to save the report
  --quiet                  # don't stream the report to the terminal
```

Run `research-agent --help` for the full list.

### Use it from Python

```python
import asyncio
from research_agent import Settings, run_research

result = asyncio.run(run_research("How do solid-state batteries work?", Settings(max_rounds=1)))
print(result.report)
print(result.usage.summary())
```

## Cost

Every run uses `claude-opus-5` by default and makes roughly `2 + sub-questions + follow-ups` model calls, plus web searches ($10 per 1,000 searches). The terminal prints token and search counts at the end of each run. To make runs cheaper, lower `--sub-questions`, `--rounds`, `--max-searches` or `--research-effort`, or pass `--model claude-sonnet-5`.

**Refusal fallbacks:** requests are sent with server-side fallbacks enabled (`fallbacks: "default"`). If Claude's safety classifiers decline a request, the API retries it on Anthropic's recommended fallback model instead of failing. Pass `--no-fallbacks` to turn this off.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests run offline against fake clients, so they need no API key and cost nothing.

```
research_agent/
  cli.py           command-line entry point
  orchestrator.py  runs the team and the critic loop
  planner.py       planner agent
  researcher.py    researcher agent (web search/fetch)
  critic.py        critic agent
  writer.py        writer agent
  sources.py       citation extraction and global source numbering
  models.py        schemas and data classes
  llm.py           shared request options, stop-reason checks, usage tracking
  config.py        Settings
```

## Ideas for extending it

- Add a local-files researcher, so the team can research your own PDFs and notes alongside the web.
- Cache reports and reuse earlier findings for related questions.
- Put a small web UI in front of it, or run it on a schedule for a daily digest on a topic.
