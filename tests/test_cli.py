import json

import pytest
from click.testing import CliRunner

from synthesis_deliberator.cli import main
from synthesis_deliberator.models import AI_PARADIGM, EXPORT_SCHEMA

from test_engine import doc, good_evidence, good_interps


@pytest.fixture
def full(tmp_path):
    p = tmp_path / "full.json"
    p.write_text(json.dumps(doc()))
    return p


@pytest.fixture
def partial(tmp_path):
    p = tmp_path / "partial.json"
    p.write_text(json.dumps(doc(interps=good_interps()[:1])))
    return p


def invoke(*args):
    try:  # click < 8.2 mixes stderr into output unless told not to; 8.2 always separates
        runner = CliRunner(mix_stderr=False)
    except TypeError:
        runner = CliRunner()
    return runner.invoke(main, [str(a) for a in args])


def test_paradigm_command():
    r = invoke("paradigm")
    assert r.exit_code == 0
    assert AI_PARADIGM in r.output


def test_version():
    assert invoke("--version").exit_code == 0


def test_run_no_llm_json_is_valid_and_versioned(full):
    r = invoke("run", full, "--no-llm", "--format", "json")
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert payload["schema"] == EXPORT_SCHEMA
    assert payload["report"]["model"] is None
    assert "_cpr_synthetic" not in payload


def test_run_text(partial):
    r = invoke("run", partial, "--no-llm")
    assert r.exit_code == 0
    assert "PREMATURE_CONVERGENCE" in r.output
    assert "Deliberation prompts" in r.output


def test_threshold_options_change_findings(tmp_path):
    p = tmp_path / "thin.json"
    p.write_text(json.dumps(doc(evidence=good_evidence()[:5])))
    assert "THIN_EVIDENCE" in invoke("run", p, "--no-llm").output
    assert "THIN_EVIDENCE" not in invoke("run", p, "--no-llm", "--min-evidence", "5").output


def test_threshold_options_are_documented_with_defaults():
    help_text = invoke("run", "--help").output
    for opt, default in (("--min-evidence", "6"), ("--dominance", "0.5"),
                         ("--overlap-support", "0.7"), ("--overlap-statement", "0.6")):
        assert opt in help_text
        assert f"default: {default}" in help_text


def test_export_to_file(full, tmp_path):
    out = tmp_path / "out.json"
    r = invoke("export", full, "--no-llm", "-o", out)
    assert r.exit_code == 0
    assert json.loads(out.read_text())["schema"] == EXPORT_SCHEMA


def test_matrix_on_input_and_export(full, tmp_path):
    r = invoke("matrix", full)
    assert r.exit_code == 0
    assert r.output.splitlines()[0].split()[:4] == ["evidence", "I1", "I2", "I3"]
    out = tmp_path / "out.json"
    invoke("export", full, "--no-llm", "-o", out)
    assert invoke("matrix", out).output == r.output


def test_settle_writes_new_export_and_never_touches_source(full, tmp_path):
    before = full.read_bytes()
    out = tmp_path / "settled.json"
    r = invoke("settle", full, "--choose", "I1", "--rationale", "Permissions came up most",
               "--settled-at", "2026-11-01T00:00:00+00:00", "-o", out)
    assert r.exit_code == 0
    assert full.read_bytes() == before
    data = json.loads(out.read_text())
    assert data["schema"] == EXPORT_SCHEMA
    assert data["report"]["settlement"] == {"chosen": "I1", "rationale": "Permissions came up most",
                                            "settled_at": "2026-11-01T00:00:00+00:00"}
    assert "UNADDRESSED_STRAIN" in r.stderr


def test_settle_refuses_to_overwrite_source(full):
    before = full.read_bytes()
    r = invoke("settle", full, "--choose", "open", "--rationale", "x", "-o", full)
    assert r.exit_code != 0
    assert full.read_bytes() == before


def test_settle_on_export_keeps_its_interpretations(full, tmp_path):
    exp = tmp_path / "exp.json"
    invoke("export", full, "--no-llm", "-o", exp)
    r = invoke("settle", exp, "--choose", "open", "--rationale", "need funnel data")
    assert r.exit_code == 0
    report = json.loads(r.stdout)["report"]
    assert [i["id"] for i in report["interpretations"]] == ["I1", "I2", "I3"]
    assert report["settlement"]["chosen"] == "open"


def test_settle_rejects_unknown_id(full):
    r = invoke("settle", full, "--choose", "I9", "--rationale", "x")
    assert r.exit_code != 0
    assert "I9" in r.stderr


def test_bad_input_is_a_clean_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"evidence": "nope"}))
    r = invoke("run", p, "--no-llm")
    assert r.exit_code != 0
    assert "Traceback" not in r.output


def test_run_with_model_down_degrades_visibly(partial, monkeypatch):
    from synthesis_deliberator import llm

    def boom(prompt, model):
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "_call_model", boom)
    r = invoke("run", partial, "--model", "any/model")
    assert r.exit_code == 0
    assert "model unavailable (any/model)" in r.output
