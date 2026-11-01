"""Click CLI. The `paradigm` command is mandatory on every CPR Phase II tool."""

from __future__ import annotations

import json
import sys

import click

from . import __version__
from .engine import analyze
from .io import load_input, render_text, to_export
from .models import AI_PARADIGM, AI_PARADIGM_RATIONALE, TOOL_NAME


def _run(path: str, no_llm: bool, model: str | None):
    report = analyze(load_input(path))
    if not no_llm:
        from .llm import enrich

        report.enrichment = enrich(report, model=model)
    return report


@click.group()
@click.version_option(__version__, prog_name=TOOL_NAME)
def main() -> None:
    """TEMPLATE: one-line description of the tool."""


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--no-llm", is_flag=True, help="Rule-based core only; never calls a model.")
@click.option("--model", default=None, help="litellm model id (or set CPR_LLM_MODEL).")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def run(path: str, no_llm: bool, model: str | None, fmt: str) -> None:
    """Analyse an input file."""
    report = _run(path, no_llm, model)
    if fmt == "json":
        click.echo(json.dumps(to_export(report), indent=2))
    else:
        click.echo(render_text(report))


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None)
@click.option("--no-llm", is_flag=True)
@click.option("--model", default=None)
def export(path: str, output: str | None, no_llm: bool, model: str | None) -> None:
    """Write the versioned JSON export."""
    payload = json.dumps(to_export(_run(path, no_llm, model)), indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        click.echo(f"Wrote {output}", err=True)
    else:
        sys.stdout.write(payload + "\n")


@main.command()
def paradigm() -> None:
    """Print the AI paradigm and why it fits this research task."""
    click.echo(f"AI paradigm: {AI_PARADIGM}\n")
    click.echo(AI_PARADIGM_RATIONALE)


if __name__ == "__main__":
    main()
