import json

import pytest

from synthesis_deliberator import llm
from synthesis_deliberator.engine import analyze

from test_engine import doc, good_interps


def reply(*interps):
    return json.dumps({"interpretations": list(interps)})


M_A = {"statement": "Setup stalls waiting on IT to approve SSO", "mechanism": "gatekeeping",
       "supports": ["E3", "E6"], "strains": ["E2"], "assumptions": ["IT owns identity"],
       "discriminator": "SSO step dwell time", "confidence": 0.8, "rank": 1}
M_B = {"statement": "Trial users were never meant to finish setup", "supports": ["E4"],
       "strains": ["E1"], "discriminator": "Intent captured at signup"}
M_C = {"statement": "Pricing page promises a feature setup does not show", "supports": ["E5"],
       "strains": ["E3"], "discriminator": "Referrer is pricing page"}


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv(llm.MODEL_ENV, raising=False)


def test_model_precedence(monkeypatch):
    assert llm.resolve_model() == llm.DEFAULT_MODEL == "anthropic/claude-opus-5-5"
    monkeypatch.setenv(llm.MODEL_ENV, "openai/gpt-x")
    assert llm.resolve_model() == "openai/gpt-x"
    assert llm.resolve_model("ollama/llama3.1") == "ollama/llama3.1"


def test_model_writes_all_three_when_none_supplied(monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda p, m: reply(M_A, M_B, M_C))
    r = llm.enrich(analyze(doc(interps=[])), model="any/model")
    assert [(i.id, i.origin) for i in r.interpretations] == [("M1", "model"), ("M2", "model"), ("M3", "model")]
    assert r.model == "any/model"
    assert "PREMATURE_CONVERGENCE" not in [f.rule_id for f in r.findings]
    assert r.prompts == []


def test_model_completes_set_and_never_edits_researcher_readings(monkeypatch):
    seen = {}

    def fake(prompt, model):
        seen["prompt"] = prompt
        # A hostile reply: tries to reuse the researcher's id and add extras.
        return reply({**M_A, "id": "I1"}, M_B, M_C)

    monkeypatch.setattr(llm, "_call_model", fake)
    base = analyze(doc(interps=good_interps()[:1]))
    before = base.to_dict()
    r = llm.enrich(base, model="any/model")
    assert base.to_dict() == before
    assert r.interpretations[0] == base.interpretations[0]
    assert [(i.id, i.origin) for i in r.interpretations] == [
        ("I1", "researcher"), ("M1", "model"), ("M2", "model")]
    assert "Write 2 more interpretations" in seen["prompt"]
    assert "do not restate, edit or improve them" in seen["prompt"]


def test_model_not_called_when_set_complete(monkeypatch):
    def boom(prompt, model):
        raise AssertionError("should not be called")

    monkeypatch.setattr(llm, "_call_model", boom)
    r = llm.enrich(analyze(doc()))
    assert r.model is None
    assert "model not called" in r.model_note


def test_model_confidence_and_rank_dropped(monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda p, m: reply(M_A, M_B, M_C))
    d = repr(llm.enrich(analyze(doc(interps=[])), model="x").to_dict())
    assert "confidence" not in d and "'rank'" not in d


def test_model_readings_are_verified_by_engine(monkeypatch):
    bad = {**M_B, "supports": ["E42"]}
    monkeypatch.setattr(llm, "_call_model", lambda p, m: reply(M_A, bad, M_C))
    r = llm.enrich(analyze(doc(interps=[])), model="x")
    assert ("UNANCHORED", "M2") in [(f.rule_id, f.location) for f in r.findings]
    assert {c.interpretation_id: c.status for c in r.checks}["M2"] == "unanchored"


def test_prompt_forbids_personas_invention_and_verdicts():
    p = llm.build_prompt(analyze(doc(interps=[])), 3)
    for phrase in ("not personas", "Never invent evidence", "Do not say which interpretation is true",
                   "No confidence"):
        assert phrase in p


@pytest.mark.parametrize("failure", [
    lambda p, m: (_ for _ in ()).throw(RuntimeError("backend down")),
    lambda p, m: "I cannot help with that.",
    lambda p, m: '{"interpretations": "nope"}',
    lambda p, m: reply({"statement": ""}),
])
def test_model_failure_equals_no_llm_report_apart_from_note(monkeypatch, failure):
    monkeypatch.setattr(llm, "_call_model", failure)
    base = analyze(doc(interps=good_interps()[:1]))
    r = llm.enrich(base, model="any/model")
    got, want = r.to_dict(), base.to_dict()
    assert got.pop("model_note").startswith("model unavailable (any/model)")
    want.pop("model_note")
    assert got == want
