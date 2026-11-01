"""Data models and paradigm constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

TOOL_NAME = "cpr-tool"  # TEMPLATE: set to the tool slug
EXPORT_SCHEMA = "cpr.tool.export/1"  # TEMPLATE: cpr.<abbrev>.export/1

AI_PARADIGM = "Rule-based"  # TEMPLATE: the roadmap's primary paradigm
AI_PARADIGM_RATIONALE = (
    # TEMPLATE: one paragraph on why this paradigm fits this research task.
    "This skeleton uses a deterministic rule-based core so that every finding is "
    "reproducible and auditable without a model. Replace this paragraph with the "
    "methodological case for the tool's own paradigm."
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
