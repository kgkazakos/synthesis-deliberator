"""synthesis-deliberator: CPR Phase II, Frame A (Test-Time Deliberation), tool 3.

A reasoning model holds three competing interpretations of research evidence open. The
deterministic core (--no-llm) checks evidence readiness, verifies every interpretation
against the evidence, builds the diagnosticity-ordered matrix and checks the researcher's
settlement. It never ranks interpretations or settles on one.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .engine import analyze
from .models import AI_PARADIGM, AI_PARADIGM_RATIONALE, EXPORT_SCHEMA, Finding, Interpretation, Report

__all__ = [
    "__version__",
    "AI_PARADIGM",
    "AI_PARADIGM_RATIONALE",
    "EXPORT_SCHEMA",
    "Finding",
    "Interpretation",
    "Report",
    "analyze",
]
