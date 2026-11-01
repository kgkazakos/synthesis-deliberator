"""Deterministic core. Runs with no model (--no-llm).

Three layers, all rule-based:
1. Readiness: is the evidence set fit to synthesise from?
2. Verifier: is each interpretation (researcher or model) anchored in that evidence,
   distinct from the others, and honest about what it struggles to explain?
3. Settlement check: does the researcher's settlement address the evidence the chosen
   reading strains against?

Plus the evidence x interpretation matrix, ordered by diagnosticity (ACH). The engine
never ranks, scores or selects an interpretation.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable, Optional

from .models import (
    DIAGNOSTICITY,
    GROUNDING_FACTORS,
    SYNTHETIC_KEY,
    SYNTHETIC_THRESHOLD,
    TARGET_COUNT,
    TOOL_NAME,
    EvidenceItem,
    Finding,
    Interpretation,
    InterpretationCheck,
    MatrixRow,
    Report,
    Settlement,
    Thresholds,
)

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_STAGE_ORDER = {"readiness": 0, "verifier": 1, "settlement": 2}

_STOPWORDS = frozenset(
    "the and for are but not you all any can had her was one our out has have this that with "
    "they them their there from were been will would could should into than then when what "
    "which while who whom why how its it's also more most some such only own same too very "
    "just because about after before over under again further once these those does did doing "
    "being both each few other nor off per via".split()
)


# --------------------------------------------------------------------------- parsing


def syntheticity(entry: Any, _path: frozenset[int] = frozenset()) -> float:
    """Share of an artifact that is synthetic, read recursively from `_cpr_synthetic`.

    Human source (no block) -> 0.0. Synthetic with no ancestry -> 1.0. Otherwise
    g * mean(parent syntheticity) + (1 - g), with g the human_grounding factor.
    A cycle in the ancestry counts as fully synthetic.
    """
    block = entry.get(SYNTHETIC_KEY) if isinstance(entry, dict) and SYNTHETIC_KEY in entry else entry
    if not isinstance(block, dict) or not any(k in block for k in ("tool", "ancestry", "human_grounding", "model")):
        return 0.0
    if id(entry) in _path:
        return 1.0
    ancestry = block.get("ancestry") or []
    if not isinstance(ancestry, list):
        ancestry = [ancestry]
    if not ancestry:
        return 1.0
    g = GROUNDING_FACTORS.get(str(block.get("human_grounding", "none")).strip().lower(), 0.0)
    path = _path | {id(entry)}
    parent_mean = sum(syntheticity(p, path) for p in ancestry) / len(ancestry)
    return round(max(0.0, min(1.0, g * parent_mean + (1.0 - g))), 6)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        value = [value]
    return tuple(dict.fromkeys(_text(v) for v in value if _text(v)))


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        value = [value]
    return tuple(_text(v) for v in value if _text(v))


def parse_evidence(raw: Any) -> tuple[list[EvidenceItem], list[int]]:
    """Return evidence items and the 1-based positions of items that had no id.

    Items without an id get a positional id `#n` so they still appear in the matrix.
    """
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        raise ValueError("'evidence' must be a list of objects.")
    items, missing = [], []
    for n, e in enumerate(raw, start=1):
        if isinstance(e, str):
            e = {"text": e}
        if not isinstance(e, dict):
            raise ValueError(f"Evidence item {n} must be an object or a string.")
        eid = _text(e.get("id"))
        if not eid:
            eid = f"#{n}"
            missing.append(n)
        items.append(EvidenceItem(
            id=eid,
            text=_text(e.get("text")),
            participant=_text(e.get("participant")) or None,
            source=_text(e.get("source") or e.get("method")) or None,
            syntheticity=syntheticity(e),
        ))
    return items, missing


def parse_interpretation(raw: dict[str, Any], default_id: str, origin: str) -> Interpretation:
    """Build an interpretation from a dict. Keeps only known fields: any confidence,
    ranking or preference a source supplies is dropped here."""
    return Interpretation(
        id=_text(raw.get("id")) or default_id,
        origin=origin,
        statement=_text(raw.get("statement")),
        mechanism=_text(raw.get("mechanism")),
        supports=_ids(raw.get("supports")),
        strains=_ids(raw.get("strains")),
        assumptions=_strings(raw.get("assumptions")),
        discriminator=_text(raw.get("discriminator")),
    )


def parse_interpretations(raw: Any) -> list[Interpretation]:
    """Interpretations from an input file are always the researcher's."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("'interpretations' must be a list of objects.")
    out = []
    for n, i in enumerate(raw, start=1):
        if not isinstance(i, dict):
            raise ValueError(f"Interpretation {n} must be an object.")
        out.append(parse_interpretation(i, f"I{n}", "researcher"))
    return out


# --------------------------------------------------------------------------- readiness


def readiness(evidence: list[EvidenceItem], missing: Iterable[int], t: Thresholds) -> list[Finding]:
    out: list[Finding] = []
    n = len(evidence)
    if n < t.min_evidence:
        out.append(Finding("THIN_EVIDENCE", "medium",
                           f"{n} evidence items; fewer than {t.min_evidence}. Three readings of thin "
                           "evidence are three guesses.", "evidence"))
    for pos in missing:
        out.append(Finding("MISSING_ID", "high",
                           f"Evidence item {pos} has no id; shown as #{pos}. Interpretations cannot cite it.",
                           f"#{pos}"))
    seen: dict[str, int] = {}
    for e in evidence:
        seen[e.id] = seen.get(e.id, 0) + 1
    for eid, c in seen.items():
        if c > 1:
            out.append(Finding("DUPLICATE_ID", "high",
                               f"Evidence id '{eid}' is used by {c} items; citations to it are ambiguous.", eid))
    with_p = [e.participant for e in evidence if e.participant]
    if len(with_p) >= 2:
        counts: dict[str, int] = {}
        for p in with_p:
            counts[p] = counts.get(p, 0) + 1
        top, top_n = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        share = top_n / len(with_p)
        if share > t.dominance:
            out.append(Finding("PARTICIPANT_DOMINANCE", "medium",
                               f"Participant {top} supplies {top_n} of {len(with_p)} attributed items "
                               f"({share:.0%}, threshold {t.dominance:.0%}).", top))
    sources = {e.source for e in evidence if e.source}
    if n >= 2 and len(sources) == 1:
        out.append(Finding("SINGLE_SOURCE", "medium",
                           f"All attributed evidence comes from one source ({next(iter(sources))}).", "evidence"))
    share = synthetic_share(evidence)
    if share > SYNTHETIC_THRESHOLD:
        out.append(Finding("SYNTHETIC_SHARE", "high",
                           f"{share:.0%} of the evidence is synthetic (threshold {SYNTHETIC_THRESHOLD:.0%}).",
                           "evidence"))
    return out


def synthetic_share(items: Iterable[EvidenceItem]) -> float:
    items = list(items)
    if not items:
        return 0.0
    return round(sum(e.syntheticity for e in items) / len(items), 6)


# --------------------------------------------------------------------------- verifier


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def verify(interps: list[Interpretation], evidence: list[EvidenceItem],
           t: Thresholds) -> tuple[list[Finding], list[InterpretationCheck]]:
    by_id = {e.id: e for e in evidence}
    out: list[Finding] = []
    checks: list[InterpretationCheck] = []

    k = len(interps)
    if k < TARGET_COUNT:
        held = "No interpretations are" if k == 0 else f"Only {k} interpretation{'s are' if k > 1 else ' is'}"
        out.append(Finding("PREMATURE_CONVERGENCE", "high",
                           f"{held} held open; the method needs {TARGET_COUNT} competing readings "
                           "before anyone settles.", "interpretations", "verifier"))
    elif k > TARGET_COUNT:
        out.append(Finding("EXTRA_INTERPRETATIONS", "low",
                           f"{k} interpretations; the method asks for {TARGET_COUNT}. Not a failure, "
                           "but check whether some are variants of one reading.", "interpretations", "verifier"))

    for i in interps:
        cited = tuple(dict.fromkeys(i.supports + i.strains))
        unknown = tuple(c for c in cited if c not in by_id)
        known_support = [by_id[s] for s in i.supports if s in by_id]
        if unknown:
            out.append(Finding("UNANCHORED", "high",
                               f"{i.id} ({i.origin}) cites evidence that does not exist: {', '.join(unknown)}.",
                               i.id, "verifier"))
        elif not known_support:
            out.append(Finding("UNANCHORED", "high",
                               f"{i.id} ({i.origin}) cites no supporting evidence.", i.id, "verifier"))
        both = [s for s in i.supports if s in i.strains]
        if both:
            out.append(Finding("DOUBLE_CITED", "low",
                               f"{i.id} cites {', '.join(both)} as both supporting and straining it.",
                               i.id, "verifier"))
        if not i.strains:
            out.append(Finding("NO_STRAIN", "medium",
                               f"{i.id} ({i.origin}) names no evidence it struggles to explain.", i.id, "verifier"))
        if not i.discriminator:
            out.append(Finding("NO_DISCRIMINATOR", "medium",
                               f"{i.id} ({i.origin}) names no observation that would tell it apart "
                               "from the other readings.", i.id, "verifier"))
        voices = {e.participant for e in known_support if e.participant}
        if len(voices) == 1 and all(e.participant for e in known_support):
            out.append(Finding("SINGLE_VOICE_SUPPORT", "medium",
                               f"All of {i.id}'s support comes from participant {next(iter(voices))}.",
                               i.id, "verifier"))
        s_share = synthetic_share(known_support)
        if s_share > SYNTHETIC_THRESHOLD:
            out.append(Finding("SYNTHETIC_SUPPORT", "high",
                               f"{s_share:.0%} of {i.id}'s support is synthetic "
                               f"(threshold {SYNTHETIC_THRESHOLD:.0%}).", i.id, "verifier"))
        checks.append(InterpretationCheck(
            interpretation_id=i.id,
            status="unanchored" if (unknown or not known_support) else "anchored",
            unknown_ids=unknown,
            participants=len(voices),
            synthetic_share=s_share,
        ))

    for a_idx, a in enumerate(interps):
        for b in interps[a_idx + 1:]:
            sup = jaccard(a.supports, b.supports) if a.supports and b.supports else 0.0
            stm = jaccard(_tokens(a.statement), _tokens(b.statement))
            reasons = []
            if sup > t.overlap_support:
                reasons.append(f"supporting evidence overlaps {sup:.0%} (threshold {t.overlap_support:.0%})")
            if stm > t.overlap_statement:
                reasons.append(f"statements share {stm:.0%} of their content words "
                               f"(threshold {t.overlap_statement:.0%})")
            if reasons:
                out.append(Finding("NEAR_DUPLICATE", "medium",
                                   f"{a.id} ({a.origin}) and {b.id} ({b.origin}) may be one reading: "
                                   + "; ".join(reasons) + ".", f"{a.id}~{b.id}", "verifier"))

    if interps:
        cited_all = {c for i in interps for c in i.supports + i.strains}
        orphans = [e.id for e in evidence if e.id not in cited_all]
        if orphans:
            out.append(Finding("ORPHANED_EVIDENCE", "low",
                               f"Cited by no interpretation: {', '.join(orphans)}.", "evidence", "verifier"))
    return out, checks


# --------------------------------------------------------------------------- matrix


def _mark(i: Interpretation, eid: str) -> str:
    s, t = eid in i.supports, eid in i.strains
    return "both" if s and t else "supports" if s else "strains" if t else "not_cited"


def matrix(interps: list[Interpretation], evidence: list[EvidenceItem]) -> list[MatrixRow]:
    """One row per evidence item, ordered by diagnosticity, then input order.

    diagnostic: supports one reading and strains another (or both within one)
    partial:    cited by some readings, without that split
    uncited:    cited by no reading
    consistent: supports every reading and strains none; it cannot tell them apart
    """
    rows = []
    for e in evidence:
        marks = {i.id: _mark(i, e.id) for i in interps}
        vals = list(marks.values())
        sup = any(v in ("supports", "both") for v in vals)
        stn = any(v in ("strains", "both") for v in vals)
        if sup and stn:
            d = "diagnostic"
        elif vals and all(v == "supports" for v in vals):
            d = "consistent"
        elif sup or stn:
            d = "partial"
        else:
            d = "uncited"
        rows.append(MatrixRow(evidence_id=e.id, marks=marks, diagnosticity=d))
    order = {d: n for n, d in enumerate(DIAGNOSTICITY)}
    return [r for _, r in sorted(enumerate(rows), key=lambda p: (order[p[1].diagnosticity], p[0]))]


# --------------------------------------------------------------------------- prompts


def deliberation_prompts(interps: list[Interpretation], evidence: list[EvidenceItem]) -> list[str]:
    """Templated prompts for writing the missing readings yourself (--no-llm path)."""
    missing = TARGET_COUNT - len(interps)
    if missing <= 0:
        return []
    cited = {c for i in interps for c in i.supports + i.strains}
    orphans = [e.id for e in evidence if e.id not in cited]
    strains = list(dict.fromkeys(s for i in interps for s in i.strains))
    out = []
    if strains:
        out.append(f"Start a reading from the evidence the current readings strain against "
                   f"({', '.join(strains)}): what account makes those items expected rather than awkward?")
    if orphans and interps:
        out.append(f"Start a reading from evidence no current reading cites ({', '.join(orphans)}).")
    for i in interps:
        out.append(f"Write a reading in which the mechanism behind {i.id} is absent or runs the other way, "
                   "and say what the evidence would then look like.")
    out += [
        "Write a reading where the pattern comes from who was sampled or how they were asked, "
        "not from the behaviour itself.",
        "Write a reading that explains the same evidence at a different level: individual, team, "
        "or product and context.",
        "For each reading, name one observation you could still collect that would tell it apart "
        "from the others.",
    ]
    return out[: max(missing + 1, 3)]


# --------------------------------------------------------------------------- assembly


def _sort(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (_STAGE_ORDER[f.stage], _SEVERITY_ORDER[f.severity],
                                           f.rule_id, f.location or ""))


def assemble(question: str, evidence: list[EvidenceItem], missing: list[int],
             interps: list[Interpretation], t: Thresholds) -> Report:
    from . import __version__

    found, checks = verify(interps, evidence, t)
    return Report(
        tool=TOOL_NAME,
        version=__version__,
        question=question,
        thresholds=t,
        evidence=list(evidence),
        interpretations=list(interps),
        checks=checks,
        matrix=matrix(interps, evidence),
        findings=_sort(readiness(evidence, missing, t) + found),
        prompts=deliberation_prompts(interps, evidence),
        synthetic_share=synthetic_share(evidence),
    )


def analyze(doc: dict[str, Any], thresholds: Optional[Thresholds] = None) -> Report:
    """Run readiness, verifier and matrix over an input document. Never calls a model."""
    if not isinstance(doc, dict):
        raise ValueError("Expected a JSON object with 'question' and 'evidence'.")
    evidence, missing = parse_evidence(doc.get("evidence"))
    interps = parse_interpretations(doc.get("interpretations"))
    return assemble(_text(doc.get("question")), evidence, missing, interps, thresholds or Thresholds())


def with_interpretations(report: Report, added: list[Interpretation]) -> Report:
    """Re-verify a report with interpretations appended (the model layer's entry point).

    Existing interpretations are kept exactly as they are; readiness findings are carried
    over unchanged.
    """
    interps = list(report.interpretations) + list(added)
    found, checks = verify(interps, report.evidence, report.thresholds)
    kept = [f for f in report.findings if f.stage == "readiness"]
    return replace(
        report,
        interpretations=interps,
        checks=checks,
        matrix=matrix(interps, report.evidence),
        findings=_sort(kept + found),
        prompts=deliberation_prompts(interps, report.evidence),
    )


# --------------------------------------------------------------------------- settlement


def _mentions(text: str, eid: str) -> bool:
    return re.search(rf"(?<![\w#]){re.escape(eid)}(?!\w)", text, re.IGNORECASE) is not None


def settle(report: Report, chosen: str, rationale: str, settled_at: str) -> Report:
    """Attach the researcher's settlement and check it. Returns a new report."""
    chosen = chosen.strip()
    if chosen != "open" and report.interpretation(chosen) is None:
        ids = ", ".join(i.id for i in report.interpretations) or "none"
        raise ValueError(f"'{chosen}' is not an interpretation id (have: {ids}) or 'open'.")
    rationale = (rationale or "").strip()
    out: list[Finding] = []
    if not rationale:
        out.append(Finding("NO_RATIONALE", "high", "The settlement has no rationale.", chosen, "settlement"))
    if chosen != "open":
        strains = report.interpretation(chosen).strains
        missed = [s for s in strains if not _mentions(rationale, s)]
        if missed:
            out.append(Finding("UNADDRESSED_STRAIN", "medium",
                               f"{chosen} strains against {', '.join(missed)}, which the rationale "
                               "never mentions.", chosen, "settlement"))
    kept = [f for f in report.findings if f.stage != "settlement"]
    return replace(report, settlement=Settlement(chosen, rationale, settled_at), findings=_sort(kept + out))
