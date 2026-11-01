"""Loaders, renderers and the versioned export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AI_PARADIGM, EXPORT_SCHEMA, Report


def load_input(path: str | Path) -> dict[str, Any]:
    """TEMPLATE: extend for the tool's real input formats (txt, csv, jsonl, ...)."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object at the top level.")
        return data
    return {"text": p.read_text(encoding="utf-8")}


def to_export(report: Report) -> dict[str, Any]:
    return {
        "schema": EXPORT_SCHEMA,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paradigm": AI_PARADIGM,
        # No _cpr_synthetic block: this skeleton produces no synthetic data.
        # TEMPLATE: if the tool produces synthetic data, add the Provenance Hash block here.
        "report": report.to_dict(),
    }


def render_text(report: Report) -> str:
    c = report.counts()
    out = [
        f"{report.tool} {report.version}",
        f"{len(report.findings)} findings (high {c['high']}, medium {c['medium']}, low {c['low']})",
        "",
    ]
    for f in report.findings:
        loc = f" @ {f.location}" if f.location else ""
        out.append(f"[{f.severity.upper():6}] {f.rule_id}{loc}: {f.message}")
    if report.enrichment:
        out += ["", "AI enrichment (additive, not a computed value):", report.enrichment]
    return "\n".join(out)
