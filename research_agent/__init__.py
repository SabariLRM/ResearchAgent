"""Multi-agent research assistant built on Claude."""

from .config import Settings
from .orchestrator import ResearchResult, run_research

__all__ = ["ResearchResult", "Settings", "run_research"]
__version__ = "0.1.0"
