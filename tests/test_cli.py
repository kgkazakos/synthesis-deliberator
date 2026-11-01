import json
from pathlib import Path

from click.testing import CliRunner

from cpr_tool.cli import main
from cpr_tool.models import AI_PARADIGM, EXPORT_SCHEMA

EXAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "sample.json")


def test_paradigm_command():
    r = CliRunner().invoke(main, ["paradigm"])
    assert r.exit_code == 0
    assert AI_PARADIGM in r.output


def test_run_no_llm_json_is_valid_and_versioned():
    r = CliRunner().invoke(main, ["run", EXAMPLE, "--no-llm", "--format", "json"])
    assert r.exit_code == 0
    payload = json.loads(r.output)
    assert payload["schema"] == EXPORT_SCHEMA
    assert payload["report"]["enrichment"] is None
    assert "_cpr_synthetic" not in payload  # skeleton produces no synthetic data


def test_export_to_file(tmp_path):
    out = tmp_path / "out.json"
    r = CliRunner().invoke(main, ["export", EXAMPLE, "--no-llm", "-o", str(out)])
    assert r.exit_code == 0
    assert json.loads(out.read_text())["schema"] == EXPORT_SCHEMA
