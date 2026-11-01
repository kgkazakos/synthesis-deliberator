"""AI layer for cpr-tool.

Strictly additive. If no backend is configured or a call fails, this module returns
None and the rule-based report stands unchanged. It never reads or mutates any value
the engine computed. TEMPLATE: describe what this tool's AI layer produces.
"""

from __future__ import annotations

import contextlib
import io
import os
from typing import Optional

from .models import Report

DEFAULT_MODEL = os.environ.get("CPR_LLM_MODEL", "anthropic/claude-sonnet-5-5")

# Any litellm-routable model works. Override with --model or CPR_LLM_MODEL.


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


def enrich(report: Report, model: Optional[str] = None) -> Optional[str]:
    """Return an enrichment string, or None if no model is configured or anything fails.

    TEMPLATE: replace the prompt with the tool's enrichment (trace, narrative, etc.).
    """
    model = model or DEFAULT_MODEL
    lines = [f"- [{f.severity}] {f.message}" for f in report.findings] or ["- no findings"]
    prompt = (
        "You are assisting a human researcher. Explain these rule-based findings in plain "
        "language. Do not add, remove or re-score findings.\n" + "\n".join(lines)
    )
    try:
        return _call_model(prompt, model)
    except Exception:  # graceful degradation: the rule-based result stands
        return None
