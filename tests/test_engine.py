import copy

import pytest

from synthesis_deliberator.engine import (
    analyze,
    jaccard,
    parse_interpretation,
    settle,
    syntheticity,
    with_interpretations,
)
from synthesis_deliberator.models import (
    RULES,
    Finding,
    Interpretation,
    Report,
    Thresholds,
)

SYNTH = {"_cpr_synthetic": {"tool": "x", "model": "m", "ancestry": [], "human_grounding": "none"}}


def ev(eid, participant, source="interview", text="", **extra):
    return {"id": eid, "text": text or f"quote {eid}", "participant": participant, "source": source, **extra}


def good_evidence():
    return [
        ev("E1", "P1"), ev("E2", "P2"), ev("E3", "P3"),
        ev("E4", "P4", "survey"), ev("E5", "P5", "survey"), ev("E6", "P6", "analytics"),
    ]


def good_interps():
    return [
        {"id": "I1", "statement": "Admins abandon setup because permissions are confusing",
         "supports": ["E1", "E2"], "strains": ["E5"], "discriminator": "Drop-off at the roles step"},
        {"id": "I2", "statement": "Setup is delegated to someone without budget authority",
         "supports": ["E3", "E4"], "strains": ["E1"], "discriminator": "Second user invited before abandon"},
        {"id": "I3", "statement": "Trial timing collides with quarter end procurement freezes",
         "supports": ["E5", "E6"], "strains": ["E2"], "discriminator": "Abandon rate peaks in final fortnight"},
    ]


def doc(evidence=None, interps=None):
    return {"question": "Why do first-time admins abandon setup?",
            "evidence": good_evidence() if evidence is None else evidence,
            "interpretations": good_interps() if interps is None else interps}


def rules(report, stage=None):
    return [f.rule_id for f in report.findings if stage is None or f.stage == stage]


# ------------------------------------------------------------------ clean baseline


def test_clean_input_has_no_findings():
    r = analyze(doc())
    assert r.findings == []
    assert [c.status for c in r.checks] == ["anchored"] * 3
    assert r.prompts == []


def test_every_rule_is_documented():
    import synthesis_deliberator.engine as eng
    src = open(eng.__file__).read()
    for rule in RULES:
        assert f'"{rule}"' in src


def test_deterministic():
    d = doc(interps=good_interps()[:1])
    assert analyze(d).to_dict() == analyze(copy.deepcopy(d)).to_dict()


# ------------------------------------------------------------------ readiness


def test_thin_evidence_and_threshold_option():
    assert "THIN_EVIDENCE" in rules(analyze(doc(evidence=good_evidence()[:5])))
    assert "THIN_EVIDENCE" not in rules(analyze(doc()))
    assert "THIN_EVIDENCE" in rules(analyze(doc(), Thresholds(min_evidence=7)))


def test_missing_id_gets_positional_id():
    e = good_evidence()
    del e[2]["id"]
    r = analyze(doc(evidence=e))
    assert "MISSING_ID" in rules(r)
    assert "#3" in [x.id for x in r.evidence]


def test_duplicate_id():
    e = good_evidence()
    e[1]["id"] = "E1"
    assert "DUPLICATE_ID" in rules(analyze(doc(evidence=e)))


def test_participant_dominance_and_threshold_option():
    e = good_evidence()
    for item in e[:4]:
        item["participant"] = "P1"
    r = analyze(doc(evidence=e))
    f = next(f for f in r.findings if f.rule_id == "PARTICIPANT_DOMINANCE")
    assert f.location == "P1"
    assert "PARTICIPANT_DOMINANCE" not in rules(analyze(doc(evidence=e), Thresholds(dominance=0.7)))
    assert "PARTICIPANT_DOMINANCE" not in rules(analyze(doc()))


def test_single_source():
    e = [ev(f"E{n}", f"P{n}") for n in range(1, 7)]
    assert "SINGLE_SOURCE" in rules(analyze(doc(evidence=e)))


def test_synthetic_share_reads_ancestry():
    e = good_evidence()
    for item in e[:2]:
        item.update(SYNTH)
    r = analyze(doc(evidence=e))
    assert r.synthetic_share == pytest.approx(2 / 6)
    assert "SYNTHETIC_SHARE" in rules(r)
    e2 = good_evidence()
    e2[0].update(SYNTH)
    assert "SYNTHETIC_SHARE" not in rules(analyze(doc(evidence=e2)))


def test_syntheticity_grounding_and_cycles():
    assert syntheticity({"text": "human"}) == 0.0
    assert syntheticity(SYNTH) == 1.0
    seeded = {"_cpr_synthetic": {"tool": "x", "ancestry": [{"text": "human"}], "human_grounding": "seeded"}}
    assert syntheticity(seeded) == pytest.approx(0.25)
    loop = {"_cpr_synthetic": {"tool": "x", "human_grounding": "partial", "ancestry": []}}
    loop["_cpr_synthetic"]["ancestry"].append(loop)
    assert syntheticity(loop) == 1.0


# ------------------------------------------------------------------ verifier


def test_premature_convergence_below_three_is_high():
    for k in (0, 1, 2):
        r = analyze(doc(interps=good_interps()[:k]))
        f = next(f for f in r.findings if f.rule_id == "PREMATURE_CONVERGENCE")
        assert f.severity == "high"


def test_more_than_three_is_a_low_note_not_convergence():
    extra = good_interps() + [{"id": "I4", "statement": "Pricing page sets the wrong expectation",
                               "supports": ["E6"], "strains": ["E3"], "discriminator": "x"}]
    r = analyze(doc(interps=extra))
    f = next(f for f in r.findings if f.rule_id == "EXTRA_INTERPRETATIONS")
    assert f.severity == "low"
    assert "PREMATURE_CONVERGENCE" not in rules(r)


def test_unanchored_unknown_ids_and_no_support():
    i = good_interps()
    i[0]["supports"] = ["E1", "E99"]
    i[1]["supports"] = []
    r = analyze(doc(interps=i))
    un = [f for f in r.findings if f.rule_id == "UNANCHORED"]
    assert [f.location for f in un] == ["I1", "I2"]
    assert "E99" in un[0].message
    checks = {c.interpretation_id: c for c in r.checks}
    assert checks["I1"].status == "unanchored" and checks["I1"].unknown_ids == ("E99",)
    assert checks["I3"].status == "anchored"


def test_double_cited():
    i = good_interps()
    i[0]["strains"] = ["E1"]
    assert "DOUBLE_CITED" in rules(analyze(doc(interps=i)))


def test_no_strain_and_no_discriminator():
    i = good_interps()
    i[0]["strains"] = []
    i[1]["discriminator"] = ""
    r = analyze(doc(interps=i))
    assert ("NO_STRAIN", "I1") in [(f.rule_id, f.location) for f in r.findings]
    assert ("NO_DISCRIMINATOR", "I2") in [(f.rule_id, f.location) for f in r.findings]


def test_near_duplicate_by_support_overlap_and_threshold_option():
    i = good_interps()
    i[1]["supports"] = ["E1", "E2"]
    r = analyze(doc(interps=i))
    f = next(f for f in r.findings if f.rule_id == "NEAR_DUPLICATE")
    assert f.location == "I1~I2"
    assert "NEAR_DUPLICATE" not in rules(analyze(doc(interps=i), Thresholds(overlap_support=1.0)))


def test_near_duplicate_by_statement_wording():
    i = good_interps()
    i[2]["statement"] = "Admins abandon setup because the permissions feel confusing"  # 5 of 6 content words
    assert "NEAR_DUPLICATE" in rules(analyze(doc(interps=i)))
    assert "NEAR_DUPLICATE" not in rules(analyze(doc(interps=i), Thresholds(overlap_statement=0.9)))


def test_single_voice_support():
    e = good_evidence()
    e[1]["participant"] = "P1"
    r = analyze(doc(evidence=e))
    assert ("SINGLE_VOICE_SUPPORT", "I1") in [(f.rule_id, f.location) for f in r.findings]


def test_synthetic_support():
    e = good_evidence()
    e[0].update(SYNTH)
    r = analyze(doc(evidence=e))
    assert ("SYNTHETIC_SUPPORT", "I1") in [(f.rule_id, f.location) for f in r.findings]
    assert "SYNTHETIC_SHARE" not in rules(r)  # 1 of 6 overall


def test_orphaned_evidence():
    e = good_evidence() + [ev("E7", "P7", "diary")]
    f = next(f for f in analyze(doc(evidence=e)).findings if f.rule_id == "ORPHANED_EVIDENCE")
    assert "E7" in f.message


def test_model_side_fields_are_dropped_on_parse():
    i = parse_interpretation({"statement": "s", "supports": ["E1"], "confidence": 0.9,
                              "rank": 1, "preferred": True}, "M1", "model")
    d = i.__dict__
    assert "confidence" not in d and "rank" not in d and "preferred" not in d


def test_no_ranking_or_score_anywhere_in_report():
    d = analyze(doc()).to_dict()
    flat = repr(d).lower()
    for word in ("score", "rank", "winner", "confidence", "best"):
        assert word not in flat


# ------------------------------------------------------------------ matrix


def test_matrix_orders_by_diagnosticity():
    e = good_evidence() + [ev("E7", "P7", "diary"), ev("E8", "P8", "diary")]
    i = good_interps()
    for x in i:
        x["supports"] = x["supports"] + ["E8"]
    r = analyze(doc(evidence=e, interps=i))
    order = [(m.evidence_id, m.diagnosticity) for m in r.matrix]
    assert order[-1] == ("E8", "consistent")
    assert order[-2] == ("E7", "uncited")
    diag = [eid for eid, d in order if d == "diagnostic"]
    assert diag == ["E1", "E2", "E5"]  # input order within a class
    assert ("E3", "partial") in order


def test_matrix_marks():
    i = good_interps()
    i[0]["strains"] = ["E1"]
    r = analyze(doc(interps=i))
    row = next(m for m in r.matrix if m.evidence_id == "E1")
    assert row.marks == {"I1": "both", "I2": "strains", "I3": "not_cited"}


# ------------------------------------------------------------------ prompts and model entry


def test_prompts_only_when_set_incomplete():
    assert analyze(doc(interps=good_interps()[:1])).prompts
    assert analyze(doc()).prompts == []


def test_with_interpretations_keeps_researcher_readings_and_readiness():
    base = analyze(doc(evidence=good_evidence()[:5], interps=good_interps()[:1]))
    before = list(base.interpretations)
    added = [Interpretation(id="M1", origin="model", statement="Different story", supports=("E3",),
                            strains=("E1",), discriminator="d")]
    r = with_interpretations(base, added)
    assert r.interpretations[0] is before[0]
    assert [i.origin for i in r.interpretations] == ["researcher", "model"]
    assert "THIN_EVIDENCE" in rules(r)
    assert base.interpretations == before  # original untouched


# ------------------------------------------------------------------ settlement


def test_settle_open_with_rationale_is_clean():
    r = settle(analyze(doc()), "open", "Need the roles-step funnel first.", "2026-11-01T00:00:00+00:00")
    assert r.settlement.chosen == "open"
    assert rules(r, "settlement") == []


def test_settle_no_rationale():
    r = settle(analyze(doc()), "open", "  ", "t")
    assert "NO_RATIONALE" in rules(r, "settlement")


def test_settle_unaddressed_strain():
    r = settle(analyze(doc()), "I1", "Permissions came up most.", "t")
    f = next(f for f in r.findings if f.rule_id == "UNADDRESSED_STRAIN")
    assert "E5" in f.message
    ok = settle(analyze(doc()), "I1", "E5 is a survey outlier from a reseller.", "t")
    assert "UNADDRESSED_STRAIN" not in rules(ok)


def test_settle_id_match_is_whole_word():
    i = good_interps()
    i[0]["strains"] = ["E1"]
    i[0]["supports"] = ["E2"]
    r = settle(analyze(doc(interps=i)), "I1", "E10 and E12 explain it", "t")
    assert "UNADDRESSED_STRAIN" in rules(r)


def test_settle_rejects_unknown_choice_and_does_not_mutate():
    base = analyze(doc())
    snapshot = base.to_dict()
    with pytest.raises(ValueError):
        settle(base, "I9", "x", "t")
    settle(base, "I1", "E5 addressed", "t")
    assert base.to_dict() == snapshot


def test_resettling_replaces_previous_settlement_findings():
    r = settle(analyze(doc()), "open", "", "t")
    r = settle(r, "open", "now with a reason", "t2")
    assert rules(r, "settlement") == []


# ------------------------------------------------------------------ models


def test_report_round_trips_through_dict():
    r = settle(analyze(doc(interps=good_interps()[:2])), "I1", "E5", "t")
    assert Report.from_dict(r.to_dict()).to_dict() == r.to_dict()


def test_invalid_severity_and_origin_rejected():
    with pytest.raises(ValueError):
        Finding(rule_id="X", severity="critical", message="m")
    with pytest.raises(ValueError):
        Interpretation(id="I1", origin="panel", statement="s")


def test_jaccard_edge_cases():
    assert jaccard([], []) == 0.0
    assert jaccard(["a"], ["a"]) == 1.0
