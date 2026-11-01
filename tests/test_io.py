import json

from synthesis_deliberator.engine import analyze, settle
from synthesis_deliberator.io import (
    is_export,
    load_input,
    render_matrix,
    render_text,
    report_from_export,
    to_export,
)
from synthesis_deliberator.models import EXPORT_SCHEMA

from test_engine import doc, good_interps


def test_export_is_versioned_and_has_no_synthetic_block():
    e = to_export(analyze(doc()))
    assert e["schema"] == EXPORT_SCHEMA
    assert "_cpr_synthetic" not in e
    assert e["report"]["model"] is None


def test_export_round_trip_through_json():
    r = settle(analyze(doc(interps=good_interps()[:2])), "open", "waiting", "t")
    data = json.loads(json.dumps(to_export(r)))
    assert is_export(data)
    assert report_from_export(data).to_dict() == r.to_dict()


def test_is_export_false_for_input():
    assert not is_export(doc())


def test_load_input_rejects_non_object(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("[1, 2]")
    try:
        load_input(p)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_text_leads_with_interpretations_and_shows_origin():
    t = render_text(analyze(doc()))
    assert t.index("Interpretations") < t.index("matrix") < t.index("findings")
    assert "I1 [researcher] anchored" in t
    assert "Not settled" in t


def test_text_shows_prompts_when_incomplete():
    t = render_text(analyze(doc(interps=good_interps()[:1])))
    assert "Deliberation prompts" in t
    assert "PREMATURE_CONVERGENCE" in t


def test_text_shows_settlement():
    t = render_text(settle(analyze(doc()), "I2", "E1 explained by delegation", "2026-11-01"))
    assert "Settlement (2026-11-01): I2" in t


def test_matrix_render_order_and_legend():
    m = render_matrix(analyze(doc()))
    lines = m.splitlines()
    assert lines[0].split()[:4] == ["evidence", "I1", "I2", "I3"]
    assert "Readings are not ranked" in m
    assert render_matrix(analyze(doc(interps=[]))).startswith("No interpretations")
