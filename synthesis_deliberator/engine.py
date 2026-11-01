"""Deterministic core. Runs with no model (--no-llm)."""

from __future__ import annotations

from typing import Any

from .models import TOOL_NAME, Finding, Report


def analyze(doc: dict[str, Any]) -> Report:
    """Run the rule-based engine over an input document.

    TEMPLATE: replace the placeholder rule below with the tool's real rules or algorithm.
    Keep the output deterministic: same input, same findings, same order.
    """
    from . import __version__

    findings: list[Finding] = []

    # TEMPLATE: placeholder rule. Flags empty top-level fields.
    for key in sorted(doc):
        value = doc[key]
        if value in (None, "", [], {}):
            findings.append(
                Finding(
                    rule_id="EMPTY_FIELD",
                    severity="low",
                    message=f"Field '{key}' is empty.",
                    location=key,
                )
            )

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order[f.severity], f.rule_id, f.location or ""))
    return Report(tool=TOOL_NAME, version=__version__, findings=findings)
