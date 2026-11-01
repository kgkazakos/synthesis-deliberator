"""Loaders, renderers and the versioned export."""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from .models import AI_PARADIGM, EXPORT_SCHEMA, Report

MARK_GLYPH = {"supports": "+", "strains": "-", "both": "+/-", "not_cited": "."}


def load_json(path: Union[str, Path]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_input(path: Union[str, Path]) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object at the top level.")
    return data


def is_export(data: Any) -> bool:
    return isinstance(data, dict) and data.get("schema") == EXPORT_SCHEMA and "report" in data


def report_from_export(data: dict[str, Any]) -> Report:
    return Report.from_dict(data["report"])


def to_export(report: Report) -> dict[str, Any]:
    return {
        "schema": EXPORT_SCHEMA,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paradigm": AI_PARADIGM,
        # No _cpr_synthetic block: interpretations are analytic text labelled with the model
        # id in report.model and each interpretation's origin, not synthetic research data.
        "report": report.to_dict(),
    }


def _wrap(text: str, indent: str) -> list[str]:
    return textwrap.wrap(text, width=96, initial_indent=indent, subsequent_indent=indent) or [indent.rstrip()]


def render_matrix(report: Report) -> str:
    if not report.interpretations:
        return "No interpretations yet: nothing to cross against the evidence."
    ids = [i.id for i in report.interpretations]
    w_e = max([len("evidence")] + [len(m.evidence_id) for m in report.matrix])
    w_c = max([4] + [len(i) for i in ids])
    head = f"{'evidence':<{w_e}}  " + "  ".join(f"{i:^{w_c}}" for i in ids) + "  diagnosticity"
    lines = [head, "-" * len(head)]
    for m in report.matrix:
        cells = "  ".join(f"{MARK_GLYPH[m.marks[i]]:^{w_c}}" for i in ids)
        lines.append(f"{m.evidence_id:<{w_e}}  {cells}  {m.diagnosticity}")
    lines.append("")
    lines.append("+ supports   - strains   +/- both   . not cited. Most diagnostic evidence first; "
                 "evidence consistent with every reading last. Readings are not ranked.")
    return "\n".join(lines)


def render_text(report: Report) -> str:
    c = report.counts()
    participants = len({e.participant for e in report.evidence if e.participant})
    sources = len({e.source for e in report.evidence if e.source})
    origins = [i.origin for i in report.interpretations]
    out = [
        f"{report.tool} {report.version}",
        f"Question: {report.question or '(none given)'}",
        f"Evidence: {len(report.evidence)} items, {participants} participants, {sources} sources, "
        f"synthetic share {report.synthetic_share:.0%}",
    ]
    if report.model:
        out.append(f"Model: {report.model} (model readings are additive and verified by the engine)")
    if report.model_note:
        out.append(f"Note: {report.model_note}")
    out += ["", f"Interpretations ({len(origins)}: researcher {origins.count('researcher')}, "
                f"model {origins.count('model')})"]
    checks = {k.interpretation_id: k for k in report.checks}
    flags: dict[str, list[str]] = {}
    for f in report.findings:
        if f.stage == "verifier" and f.location:
            for loc in f.location.split("~"):
                flags.setdefault(loc, []).append(f.rule_id)
    for i in report.interpretations:
        k = checks.get(i.id)
        status = k.status if k else "unchecked"
        out.append(f"  {i.id} [{i.origin}] {status}" + (f"  flags: {', '.join(flags[i.id])}" if i.id in flags else ""))
        out += _wrap(i.statement or "(no statement)", "     ")
        if i.mechanism:
            out += _wrap(f"Mechanism: {i.mechanism}", "     ")
        out.append(f"     Supports: {', '.join(i.supports) or 'none'}   Strains: {', '.join(i.strains) or 'none'}")
        for a in i.assumptions:
            out += _wrap(f"Assumes: {a}", "     ")
        out += _wrap(f"Would tell it apart: {i.discriminator or 'not stated'}", "     ")
        out.append("")
    if not report.interpretations:
        out += ["  none", ""]
    out += ["Evidence x interpretation matrix", render_matrix(report), ""]
    out.append(f"{len(report.findings)} findings (high {c['high']}, medium {c['medium']}, low {c['low']})")
    for f in report.findings:
        loc = f" @ {f.location}" if f.location else ""
        out.append(f"[{f.severity.upper():6}] {f.rule_id}{loc}: {f.message}")
    if report.prompts:
        out += ["", "Deliberation prompts (write the missing readings yourself):"]
        for p in report.prompts:
            out += _wrap(p, "  - ")
    out.append("")
    if report.settlement:
        s = report.settlement
        out.append(f"Settlement ({s.settled_at}): {s.chosen}")
        out += _wrap(f"Rationale: {s.rationale or '(none)'}", "  ")
    else:
        out.append("Not settled. Settling is yours: sdl settle <file> --choose <id>|open --rationale \"...\"")
    return "\n".join(out)

