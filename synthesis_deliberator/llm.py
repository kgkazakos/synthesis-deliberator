"""Model layer for synthesis-deliberator.

The model's interpretations are the product. One reasoning-model call completes the set to
three competing interpretations of the evidence: all three when the researcher supplied
none, or the missing ones when they supplied one or two. The model must differ from the
researcher's readings and never edits them; the engine appends the model's readings with
origin "model" and verifies every reading with the same rules. Any confidence, ranking or
preference the model returns is dropped on parse.

If the call fails or the reply is unusable, the report equals the --no-llm report apart
from a visible "model unavailable" note. This module never mutates the report it is given.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import re
from dataclasses import replace
from typing import Any, Optional

from .engine import parse_interpretation, with_interpretations
from .models import TARGET_COUNT, Interpretation, Report

DEFAULT_MODEL = "anthropic/claude-opus-5-5"  # reasoning is the product: default to the strongest reasoner
MODEL_ENV = "CPR_LLM_MODEL"


def resolve_model(model: Optional[str] = None) -> str:
    """Any litellm-routable id. Precedence: --model, then CPR_LLM_MODEL, then the default."""
    return model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL


@contextlib.contextmanager
def _silenced():
    """litellm prints banners to stdout/stderr that corrupt JSON output. Swallow them."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def _call_model(prompt: str, model: str) -> str:
    import litellm  # lazy: --no-llm paths never import it

    with _silenced():
        resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["choices"][0]["message"]["content"]


def build_prompt(report: Report, needed: int) -> str:
    evidence = [
        f"- {e.id} (participant: {e.participant or 'unspecified'}; source: {e.source or 'unspecified'}"
        f"{'; synthetic' if e.syntheticity > 0 else ''}): {e.text}"
        for e in report.evidence
    ] or ["- none supplied"]
    held = [
        f"- {i.id}: {i.statement} (supports: {', '.join(i.supports) or 'none'}; "
        f"strains: {', '.join(i.strains) or 'none'})"
        for i in report.interpretations
    ]
    if held:
        task = (
            f"The researcher already holds {len(held)} interpretation{'s' if len(held) > 1 else ''} "
            "(read-only; do not restate, edit or improve them):\n" + "\n".join(held) + "\n\n"
            f"Write {needed} more interpretation{'s' if needed > 1 else ''}. Each must be a genuinely "
            "different account of the evidence from the researcher's and from each other: a different "
            "mechanism, not the same story reworded or narrowed."
        )
    else:
        task = (
            f"Write {needed} competing interpretations of this evidence. Each must be a genuinely "
            "different account: a different mechanism, not the same story reworded or narrowed."
        )
    return (
        "You are helping a human researcher synthesise qualitative research evidence slowly. "
        "The goal is to hold several competing interpretations open before the researcher "
        "settles on one.\n\n"
        "Rules:\n"
        "- Write as a single analyst. Interpretations are hypotheses about the evidence, not "
        "personas, voices, stakeholders or opposing sides. Do not role-play.\n"
        "- Cite only evidence ids from the list below. Never invent evidence, quotes, "
        "participants, studies, statistics or citations.\n"
        "- For each interpretation, list the evidence it explains ('supports') and the evidence "
        "it struggles to explain ('strains'). Be honest about strains; every reading should have some.\n"
        "- 'discriminator' is one observation the researcher could still collect that would "
        "tell this reading apart from the others.\n"
        "- Do not say which interpretation is true, best or most likely. No confidence, "
        "probability or ranking. The researcher decides.\n\n"
        f"Research question: {report.question or 'not stated'}\n\n"
        "Evidence:\n" + "\n".join(evidence) + "\n\n" + task + "\n\n"
        "Return JSON only:\n"
        '{"interpretations": [{"statement": "...", "mechanism": "...", "supports": ["E1"], '
        '"strains": ["E4"], "assumptions": ["..."], "discriminator": "..."}]}'
    )


def parse_interpretations(text: str, needed: int, taken: set[str]) -> Optional[list[Interpretation]]:
    """Parse the model's JSON reply into model-origin interpretations with fresh ids.

    Returns None if nothing usable came back. Ids are M1, M2, ... skipping any already taken,
    so a model reading can never replace a researcher's.
    """
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    data: Any = json.loads(m.group(0))
    items = data.get("interpretations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    out: list[Interpretation] = []
    n = 0
    for item in items:
        if len(out) >= needed:
            break
        if not isinstance(item, dict) or not str(item.get("statement") or "").strip():
            continue
        n += 1
        while f"M{n}" in taken:
            n += 1
        clean = {k: v for k, v in item.items() if k != "id"}
        out.append(parse_interpretation(clean, f"M{n}", "model"))
    return out or None


def enrich(report: Report, model: Optional[str] = None) -> Report:
    """Return a copy of the report with the model's interpretations appended and verified.

    Researcher interpretations, readiness findings and the evidence are carried over
    unchanged. On any failure the result equals the --no-llm report apart from `model_note`.
    """
    model = resolve_model(model)
    base = copy.deepcopy(report)
    needed = TARGET_COUNT - len(base.interpretations)
    if needed <= 0:
        return replace(base, model_note=f"{len(base.interpretations)} interpretations supplied; "
                                        "model not called")
    try:
        taken = {i.id for i in base.interpretations}
        added = parse_interpretations(_call_model(build_prompt(base, needed), model), needed, taken)
    except Exception:  # graceful degradation: the rule-based result stands
        added = None
    if not added:
        return replace(base, model_note=f"model unavailable ({model}); showing the --no-llm report")
    return replace(with_interpretations(base, added), model=model)
