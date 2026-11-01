from cpr_tool.engine import analyze
from cpr_tool.models import Finding

import pytest


def test_empty_fields_flagged_in_stable_order():
    r = analyze({"b": "", "a": None, "c": "ok"})
    assert [f.location for f in r.findings] == ["a", "b"]
    assert all(f.rule_id == "EMPTY_FIELD" for f in r.findings)


def test_no_findings_on_complete_input():
    assert analyze({"a": "x"}).findings == []


def test_deterministic():
    doc = {"x": "", "y": [], "z": "ok"}
    assert analyze(doc).to_dict() == analyze(doc).to_dict()


def test_invalid_severity_rejected():
    with pytest.raises(ValueError):
        Finding(rule_id="X", severity="critical", message="m")
