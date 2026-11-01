from cpr_tool import llm
from cpr_tool.engine import analyze


def test_default_model_used_when_none_given(monkeypatch):
    seen = {}

    def fake(prompt, model):
        seen["model"] = model
        return "note"

    monkeypatch.setattr(llm, "_call_model", fake)
    llm.enrich(analyze({"a": ""}))
    assert seen["model"] == llm.DEFAULT_MODEL


def test_model_failure_degrades_and_leaves_findings_untouched(monkeypatch):
    def boom(prompt, model):
        raise RuntimeError("backend down")

    monkeypatch.setattr(llm, "_call_model", boom)
    report = analyze({"a": "", "b": "x"})
    before = report.to_dict()
    assert llm.enrich(report, model="any/model") is None
    assert report.to_dict() == before


def test_enrichment_is_separate_from_findings(monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda p, m: "plain-language note")
    report = analyze({"a": ""})
    findings_before = list(report.findings)
    report.enrichment = llm.enrich(report, model="any/model")
    assert report.enrichment == "plain-language note"
    assert report.findings == findings_before
