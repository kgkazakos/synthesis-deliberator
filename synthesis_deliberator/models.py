"""Data models and paradigm constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

TOOL_NAME = "synthesis-deliberator"
EXPORT_SCHEMA = "cpr.sdl.export/1"

AI_PARADIGM = "LLM (reasoning)"
AI_PARADIGM_RATIONALE = (
    "Synthesis fails most often by converging too early: the first plausible story absorbs "
    "the data and the rest is read as confirmation. Holding three genuinely different "
    "interpretations of the same evidence open, each tied to what it explains and what it "
    "struggles to explain, is open-ended abductive reasoning, so a reasoning model spending "
    "deliberate compute on it is the right fit. A deterministic layer checks every "
    "interpretation against the evidence the researcher supplied, flags readings that are "
    "the same story reworded, and never picks a winner. Settling is the researcher's act."
)

SEVERITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    message: str
    location: Optional[str] = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass
class Report:
    tool: str
    version: str
    findings: list[Finding] = field(default_factory=list)
    # Enrichment is additive and separate from computed values. None when --no-llm or on failure.
    enrichment: Optional[str] = None

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for f in self.findings if f.severity == s) for s in SEVERITIES}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["counts"] = self.counts()
        return d
