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
STAGES = ("readiness", "verifier", "settlement")
ORIGINS = ("researcher", "model")
TARGET_COUNT = 3

SYNTHETIC_KEY = "_cpr_synthetic"
SYNTHETIC_THRESHOLD = 0.30
# Same grounding factors as provenance-inspector, so shares agree across the tool chain.
GROUNDING_FACTORS: dict[str, float] = {"seeded": 0.75, "partial": 0.5, "none": 0.0}

# Matrix marks and the diagnosticity order (ACH): most diagnostic evidence first.
MARKS = ("supports", "strains", "both", "not_cited")
DIAGNOSTICITY = ("diagnostic", "partial", "uncited", "consistent")

RULES: dict[str, str] = {
    "THIN_EVIDENCE": "Fewer evidence items than the minimum",
    "MISSING_ID": "Evidence item without an id",
    "DUPLICATE_ID": "Two evidence items share an id",
    "PARTICIPANT_DOMINANCE": "One participant supplies most of the evidence",
    "SINGLE_SOURCE": "All evidence comes from one source or method",
    "SYNTHETIC_SHARE": "Synthetic share of the evidence above the Ouroboros threshold",
    "PREMATURE_CONVERGENCE": "Fewer than three interpretations held open",
    "EXTRA_INTERPRETATIONS": "More than three interpretations",
    "UNANCHORED": "Interpretation cites unknown evidence or no supporting evidence",
    "DOUBLE_CITED": "Evidence cited as both supporting and straining one interpretation",
    "NO_STRAIN": "Interpretation names no evidence it struggles to explain",
    "NO_DISCRIMINATOR": "Interpretation names no observation that would tell it apart",
    "NEAR_DUPLICATE": "Two interpretations are the same story reworded",
    "SINGLE_VOICE_SUPPORT": "Interpretation's support comes from one participant",
    "SYNTHETIC_SUPPORT": "Synthetic share of an interpretation's support above threshold",
    "ORPHANED_EVIDENCE": "Evidence cited by no interpretation",
    "NO_RATIONALE": "Settlement without a rationale",
    "UNADDRESSED_STRAIN": "Settled reading has straining evidence the rationale never mentions",
}


@dataclass(frozen=True)
class Thresholds:
    """Tunable rule thresholds. Defaults are documented in the README and CLI help."""

    min_evidence: int = 6  # THIN_EVIDENCE fires below this many items
    dominance: float = 0.5  # PARTICIPANT_DOMINANCE fires when one participant's share is above this
    overlap_support: float = 0.7  # NEAR_DUPLICATE: Jaccard of supporting evidence ids above this
    overlap_statement: float = 0.6  # NEAR_DUPLICATE: Jaccard of statement content words above this


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    text: str
    participant: Optional[str] = None
    source: Optional[str] = None
    syntheticity: float = 0.0


@dataclass(frozen=True)
class Interpretation:
    id: str
    origin: str
    statement: str
    mechanism: str = ""
    supports: tuple[str, ...] = ()
    strains: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    discriminator: str = ""

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, got {self.origin!r}")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    message: str
    location: Optional[str] = None
    stage: str = "readiness"

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")


@dataclass(frozen=True)
class InterpretationCheck:
    """Engine verdict on anchoring only. Never a judgement of whether the reading is true."""

    interpretation_id: str
    status: str  # "anchored" or "unanchored"
    unknown_ids: tuple[str, ...] = ()
    participants: int = 0
    synthetic_share: float = 0.0


@dataclass(frozen=True)
class MatrixRow:
    evidence_id: str
    marks: dict[str, str]
    diagnosticity: str


@dataclass(frozen=True)
class Settlement:
    chosen: str  # an interpretation id, or "open"
    rationale: str
    settled_at: str


@dataclass
class Report:
    tool: str
    version: str
    question: str
    thresholds: Thresholds
    evidence: list[EvidenceItem] = field(default_factory=list)
    interpretations: list[Interpretation] = field(default_factory=list)
    checks: list[InterpretationCheck] = field(default_factory=list)
    matrix: list[MatrixRow] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    synthetic_share: float = 0.0
    settlement: Optional[Settlement] = None
    # Model output is additive: the model id that produced any interpretations, or a note
    # when the model was asked and did not answer. Both None under --no-llm.
    model: Optional[str] = None
    model_note: Optional[str] = None

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for f in self.findings if f.severity == s) for s in SEVERITIES}

    def interpretation(self, iid: str) -> Optional[Interpretation]:
        return next((i for i in self.interpretations if i.id == iid), None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["counts"] = self.counts()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Report":
        """Rebuild a report from its export form (used by `settle` and `matrix` on exports)."""
        return cls(
            tool=d["tool"],
            version=d["version"],
            question=d.get("question", ""),
            thresholds=Thresholds(**d.get("thresholds", {})),
            evidence=[EvidenceItem(**e) for e in d.get("evidence", [])],
            interpretations=[
                Interpretation(**{**i, **{k: tuple(i.get(k, ())) for k in ("supports", "strains", "assumptions")}})
                for i in d.get("interpretations", [])
            ],
            checks=[
                InterpretationCheck(**{**c, "unknown_ids": tuple(c.get("unknown_ids", ()))})
                for c in d.get("checks", [])
            ],
            matrix=[MatrixRow(**m) for m in d.get("matrix", [])],
            findings=[Finding(**f) for f in d.get("findings", [])],
            prompts=list(d.get("prompts", [])),
            synthetic_share=d.get("synthetic_share", 0.0),
            settlement=Settlement(**d["settlement"]) if d.get("settlement") else None,
            model=d.get("model"),
            model_note=d.get("model_note"),
        )
