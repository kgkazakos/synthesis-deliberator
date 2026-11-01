"""Click CLI. The `paradigm` command is mandatory on every CPR Phase II tool."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .engine import analyze, settle as settle_report
from .io import is_export, load_json, render_matrix, render_text, report_from_export, to_export
from .models import AI_PARADIGM, AI_PARADIGM_RATIONALE, TOOL_NAME, Report, Thresholds

_D = Thresholds()


def threshold_options(f):
    f = click.option("--overlap-statement", type=click.FloatRange(0, 1), default=_D.overlap_statement,
                     show_default=True, help="NEAR_DUPLICATE: statement content-word overlap above this.")(f)
    f = click.option("--overlap-support", type=click.FloatRange(0, 1), default=_D.overlap_support,
                     show_default=True, help="NEAR_DUPLICATE: supporting-evidence overlap above this.")(f)
    f = click.option("--dominance", type=click.FloatRange(0, 1), default=_D.dominance, show_default=True,
                     help="PARTICIPANT_DOMINANCE: one participant's share of the evidence above this.")(f)
    f = click.option("--min-evidence", type=click.IntRange(0), default=_D.min_evidence, show_default=True,
                     help="THIN_EVIDENCE: fires below this many evidence items.")(f)
    return f


def model_options(f):
    f = click.option("--model", default=None,
                     help="litellm model id. Precedence: --model, CPR_LLM_MODEL, anthropic/claude-opus-5-5.")(f)
    f = click.option("--no-llm", is_flag=True, help="Deterministic core only; never calls a model.")(f)
    return f


def _load(path: str, t: Thresholds) -> tuple[Report, bool]:
    """Analyse an input file, or rebuild the report from an export. Returns (report, from_export)."""
    try:
        data = load_json(path)
        if is_export(data):
            return report_from_export(data), True
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object at the top level.")
        return analyze(data, t), False
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"{path}: {exc}") from exc


def _run(path: str, no_llm: bool, model: Optional[str], t: Thresholds) -> Report:
    report, from_export = _load(path, t)
    if not no_llm and not from_export:
        from .llm import enrich

        report = enrich(report, model=model)
    return report


def _write(payload: dict, output: Optional[str]) -> None:
    text = json.dumps(payload, indent=2)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
        click.echo(f"Wrote {output}", err=True)
    else:
        sys.stdout.write(text + "\n")


@click.group()
@click.version_option(__version__, prog_name=TOOL_NAME)
def main() -> None:
    """Hold three competing interpretations of research evidence open before you settle."""


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@model_options
@threshold_options
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def run(path, no_llm, model, min_evidence, dominance, overlap_support, overlap_statement, fmt) -> None:
    """Three interpretations of the evidence, each verified against it.

    PATH is an input file, or an export (re-rendered; the model is not called again).
    """
    report = _run(path, no_llm, model, Thresholds(min_evidence, dominance, overlap_support, overlap_statement))
    if fmt == "json":
        click.echo(json.dumps(to_export(report), indent=2))
    else:
        click.echo(render_text(report))


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None)
@model_options
@threshold_options
def export(path, output, no_llm, model, min_evidence, dominance, overlap_support, overlap_statement) -> None:
    """Write the versioned JSON export."""
    t = Thresholds(min_evidence, dominance, overlap_support, overlap_statement)
    _write(to_export(_run(path, no_llm, model, t)), output)


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@threshold_options
def matrix(path, min_evidence, dominance, overlap_support, overlap_statement) -> None:
    """Print the evidence x interpretation matrix only. Never calls a model.

    Run it on an export to see the matrix with model readings.
    """
    report, _ = _load(path, Thresholds(min_evidence, dominance, overlap_support, overlap_statement))
    click.echo(render_matrix(report))


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--choose", required=True, help="Interpretation id to settle on, or 'open' to keep all three open.")
@click.option("--rationale", required=True, help="Why. Name any straining evidence ids you have weighed.")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Settlement export to write (default stdout). Never the source file.")
@click.option("--settled-at", default=None, help="ISO timestamp to record (default: now, UTC).")
@threshold_options
def settle(path, choose, rationale, output, settled_at, min_evidence, dominance, overlap_support,
           overlap_statement) -> None:
    """Record your settlement as a new export. Never modifies PATH and never calls a model.

    PATH is an export (to settle on model readings) or an input file (your readings only).
    """
    if output and Path(output).resolve() == Path(path).resolve():
        raise click.ClickException("Refusing to overwrite the source file; choose another -o path.")
    report, _ = _load(path, Thresholds(min_evidence, dominance, overlap_support, overlap_statement))
    stamp = settled_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        settled = settle_report(report, choose, rationale, stamp)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _write(to_export(settled), output)
    for f in settled.findings:
        if f.stage == "settlement":
            click.echo(f"[{f.severity.upper()}] {f.rule_id}: {f.message}", err=True)


@main.command()
def paradigm() -> None:
    """Print the AI paradigm and why it fits this research task."""
    click.echo(f"AI paradigm: {AI_PARADIGM}\n")
    click.echo(AI_PARADIGM_RATIONALE)


if __name__ == "__main__":
    main()
