"""cpr-tool: CPR Phase II, Frame X (Frame Name), tool N.

TEMPLATE: two or three lines on what the tool does, what the deterministic core
computes (--no-llm), and what the AI layer adds.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .engine import analyze
from .models import AI_PARADIGM, AI_PARADIGM_RATIONALE, EXPORT_SCHEMA, Finding, Report

__all__ = [
    "__version__",
    "AI_PARADIGM",
    "AI_PARADIGM_RATIONALE",
    "EXPORT_SCHEMA",
    "Finding",
    "Report",
    "analyze",
]
