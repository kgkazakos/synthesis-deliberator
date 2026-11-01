import json
from pathlib import Path

import pytest

from synthesis_deliberator.engine import analyze
from synthesis_deliberator.io import is_export, report_from_export
from synthesis_deliberator.models import RULES

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.json"))


def report(path):
    data = json.loads(path.read_text())
    return report_from_export(data) if is_export(data) else analyze(data)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_loads(path):
    report(path)


def test_examples_exercise_every_rule():
    fired = {f.rule_id for p in EXAMPLES for f in report(p).findings}
    assert fired == set(RULES)


def test_three_readings_is_clean():
    assert analyze(json.loads((EXAMPLES[0].parent / "three-readings.json").read_text())).findings == []
