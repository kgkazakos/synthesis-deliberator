"""Streamlit UI. Run: streamlit run synthesis_deliberator/app.py"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script, so relative imports break. Use the repo root + absolute import.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synthesis_deliberator import __version__  # noqa: E402
from synthesis_deliberator.engine import analyze, settle  # noqa: E402
from synthesis_deliberator.io import MARK_GLYPH, to_export  # noqa: E402
from synthesis_deliberator.llm import DEFAULT_MODEL, resolve_model  # noqa: E402  (litellm loads lazily)
from synthesis_deliberator.models import (  # noqa: E402
    AI_PARADIGM,
    AI_PARADIGM_RATIONALE,
    RULES,
    SYNTHETIC_THRESHOLD,
    TOOL_NAME,
    Thresholds,
)

ACCENT = "#4f3d8f"  # Frame A violet, keep in sync with .streamlit/config.toml
CHIP = {"high": "#9b2c2c", "medium": "#b7791f", "low": "#4a5568"}
STATUS_CHIP = {"anchored": "#2f6b4f", "unanchored": "#7a4b2a"}
ORIGIN_CHIP = {"researcher": "#3d3a35", "model": ACCENT}
DIAG_COLOR = {"diagnostic": ACCENT, "partial": "#5a554c", "uncited": "#7a4b2a", "consistent": "#9a948a"}
EXAMPLES = sorted((_ROOT / "examples").glob("*.json"))
# litellm ids, checked against litellm's model list. Each needs its provider key in the environment.
MODEL_PRESETS = {
    "Anthropic · Claude Opus 5.5 (default)": DEFAULT_MODEL,
    "Anthropic · Claude Sonnet 5.5": "anthropic/claude-sonnet-5-5",
    "OpenAI · GPT-5.5": "openai/gpt-5.5",
    "Google · Gemini 3.5 Flash": "gemini/gemini-3.5-flash",
    "Local · Ollama llama3.1": "ollama/llama3.1",
    "Other (type a litellm id)": None,
}
_D = Thresholds()

st.set_page_config(page_title=TOOL_NAME, layout="wide")
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
code, .mono {{ font-family: 'DM Mono', monospace; }}
.stApp {{ background: #fbfaf8; }}
section[data-testid="stSidebar"] {{ background: #16161a; }}
section[data-testid="stSidebar"] * {{ color: #ece9e2 !important; }}
div[data-testid="column"]:first-child > div {{ background: #f2ede1; border-radius: 10px; padding: 8px 12px; }}
.card {{ background: #fff; border: 1px solid #e8e2d6; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }}
.reading {{ border-top: 4px solid {ACCENT}; min-height: 100%; }}
.reading.researcher {{ border-top-color: #3d3a35; }}
.chip {{ display: inline-block; padding: 1px 8px; border-radius: 999px; color: #fff; font-size: 12px; margin: 0 6px 4px 0; }}
.muted {{ color: #7a746a; font-size: 13px; }}
.field {{ margin-top: 6px; font-size: 14px; }}
.accent {{ color: {ACCENT}; }}
table.mx {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 14px; }}
table.mx th, table.mx td {{ border-bottom: 1px solid #ece6da; padding: 6px 10px; text-align: center; }}
table.mx td:first-child, table.mx th:first-child {{ text-align: left; }}
table.mx td.txt {{ text-align: left; color: #5a554c; max-width: 420px; }}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"### {TOOL_NAME}")
    st.caption(f"v{__version__} · {AI_PARADIGM}")
    with st.expander("Why this paradigm"):
        st.write(AI_PARADIGM_RATIONALE)
    st.markdown("#### Model")
    use_ai = st.toggle("Complete the set to three (model)", value=True,
                       help="Off is the --no-llm report: readiness, verifier and matrix on your readings alone.")
    env_model = resolve_model(None)
    labels = list(MODEL_PRESETS)
    start = next((i for i, k in enumerate(labels) if MODEL_PRESETS[k] == env_model), len(labels) - 1)
    preset = st.selectbox("Preset", labels, index=start, disabled=not use_ai)
    custom = st.text_input("litellm model id", value="" if MODEL_PRESETS[preset] else env_model,
                           placeholder="provider/model", disabled=not use_ai or MODEL_PRESETS[preset] is not None)
    model = MODEL_PRESETS[preset] or custom.strip() or None
    st.caption(f"Using `{resolve_model(model)}`. Precedence: picker, then CPR_LLM_MODEL, then {DEFAULT_MODEL}.")
    with st.expander("Rules (the verifier)"):
        for rid, name in RULES.items():
            st.markdown(f"`{rid}` {name}")
    st.caption("The tool holds readings open. It never says which one is true; settling is yours.")

controls, main = st.columns([1, 2.6])
with controls:
    st.markdown("#### Input")
    names = [p.name for p in EXAMPLES]
    choice = st.selectbox("Example", names) if names else None
    default = (_ROOT / "examples" / choice).read_text(encoding="utf-8") if choice else '{"evidence": []}'
    raw = st.text_area("Question, evidence, your readings (JSON)", value=default, height=340, key=f"raw-{choice}")
    st.markdown("#### Thresholds")
    min_evidence = st.number_input("Minimum evidence items", 0, 500, _D.min_evidence)
    dominance = st.slider("Participant dominance above", 0.0, 1.0, _D.dominance, 0.05)
    overlap_support = st.slider("Near-duplicate: support overlap above", 0.0, 1.0, _D.overlap_support, 0.05)
    overlap_statement = st.slider("Near-duplicate: wording overlap above", 0.0, 1.0, _D.overlap_statement, 0.05)
    thresholds = Thresholds(int(min_evidence), dominance, overlap_support, overlap_statement)


@st.cache_data(show_spinner=False)
def deliberated(raw_json: str, t: Thresholds, model_id):
    """Cached so a rerun with unchanged inputs does not call the model again."""
    from synthesis_deliberator.llm import enrich

    return enrich(analyze(json.loads(raw_json), t), model=model_id)


def chip(label: str, color: str) -> str:
    return f"<span class='chip' style='background:{color}'>{html.escape(label)}</span>"


def esc(text: str) -> str:
    return html.escape(text or "")


with main:
    try:
        report = analyze(json.loads(raw), thresholds)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if use_ai and len(report.interpretations) < 3:
        with st.spinner(f"Deliberating with {resolve_model(model)}..."):
            report = deliberated(raw, thresholds, model)

    st.markdown("<h3 class='accent'>Three interpretations</h3>", unsafe_allow_html=True)
    if report.model_note and report.model_note.startswith("model unavailable"):
        st.warning(f"{report.model_note.capitalize()}. Check the model id and its provider key. "
                   "Your readings and the checks below are unchanged.")
    participants = len({e.participant for e in report.evidence if e.participant})
    share_note = f"synthetic {report.synthetic_share:.0%}"
    st.caption(f"{esc(report.question) or 'No question given'} · {len(report.evidence)} evidence items, "
               f"{participants} participants · {share_note}"
               + (f" · model readings by {report.model} (additive, verified by the engine)" if report.model else ""))

    flags: dict[str, list] = {}
    for f in report.findings:
        if f.stage == "verifier" and f.location:
            for loc in f.location.split("~"):
                flags.setdefault(loc, []).append(f)
    checks = {k.interpretation_id: k for k in report.checks}

    interps = report.interpretations
    for row_start in range(0, max(len(interps), 1), 3):
        row = interps[row_start:row_start + 3]
        if not row:
            break
        cols = st.columns(3)
        for col, i in zip(cols, row):
            k = checks.get(i.id)
            chips = chip(i.origin, ORIGIN_CHIP[i.origin]) + (chip(k.status, STATUS_CHIP[k.status]) if k else "")
            if k and k.synthetic_share > SYNTHETIC_THRESHOLD:
                chips += chip(f"synthetic {k.synthetic_share:.0%}", CHIP["high"])
            flag_html = "".join(chip(f.rule_id, CHIP[f.severity]) for f in flags.get(i.id, []))
            assumptions = "".join(f"<div class='field'><span class='muted'>Assumes</span> {esc(a)}</div>"
                                  for a in i.assumptions)
            col.markdown(
                f"<div class='card reading {i.origin}'><span class='mono'>{esc(i.id)}</span> {chips}"
                f"<div class='field'><b>{esc(i.statement) or '(no statement)'}</b></div>"
                + (f"<div class='field'><span class='muted'>Mechanism</span> {esc(i.mechanism)}</div>"
                   if i.mechanism else "")
                + f"<div class='field'><span class='muted'>Supports</span> "
                  f"<span class='mono'>{esc(', '.join(i.supports)) or 'none'}</span></div>"
                  f"<div class='field'><span class='muted'>Strains</span> "
                  f"<span class='mono'>{esc(', '.join(i.strains)) or 'none'}</span></div>"
                + assumptions
                + f"<div class='field'><span class='muted'>Would tell it apart</span> "
                  f"{esc(i.discriminator) or 'not stated'}</div>"
                + (f"<div class='field'>{flag_html}</div>" if flag_html else "")
                + "</div>",
                unsafe_allow_html=True,
            )
    if not interps:
        st.info("No interpretations yet. Turn the model on, or write your own using the prompts below.")
    if report.prompts:
        st.markdown("**Deliberation prompts** <span class='muted'>write the missing readings yourself</span>",
                    unsafe_allow_html=True)
        for p in report.prompts:
            st.markdown(f"- {p}")

    vf = [f for f in report.findings if f.stage == "verifier" and f.rule_id in
          ("PREMATURE_CONVERGENCE", "EXTRA_INTERPRETATIONS", "NEAR_DUPLICATE", "ORPHANED_EVIDENCE")]
    for f in vf:
        st.markdown(f"<div class='card'>{chip(f.severity, CHIP[f.severity])}<span class='mono'>{f.rule_id}"
                    f"</span> {esc(f.message)}</div>", unsafe_allow_html=True)

    if interps:
        st.markdown("#### Evidence x interpretation")
        st.caption("Most diagnostic evidence first; evidence consistent with every reading last. "
                   "+ supports, - strains, +/- both, . not cited. Readings are not ranked.")
        text = {e.id: e.text for e in report.evidence}
        head = "".join(f"<th class='mono'>{esc(i.id)}</th>" for i in interps)
        body = "".join(
            f"<tr><td class='mono'>{esc(m.evidence_id)}</td><td class='txt'>{esc(text.get(m.evidence_id, ''))}</td>"
            + "".join(f"<td class='mono'>{MARK_GLYPH[m.marks[i.id]]}</td>" for i in interps)
            + f"<td style='color:{DIAG_COLOR[m.diagnosticity]}'>{m.diagnosticity}</td></tr>"
            for m in report.matrix
        )
        st.markdown(f"<table class='mx'><tr><th>evidence</th><th></th>{head}<th>diagnosticity</th></tr>"
                    f"{body}</table>", unsafe_allow_html=True)

    readiness = [f for f in report.findings if f.stage == "readiness"]
    with st.expander(f"Evidence readiness ({len(readiness)} findings)", expanded=bool(readiness)):
        if not readiness:
            st.write("No readiness findings.")
        for f in readiness:
            st.markdown(f"<div class='card'>{chip(f.severity, CHIP[f.severity])}<span class='mono'>{f.rule_id}"
                        f"</span> {esc(f.message)}</div>", unsafe_allow_html=True)

    st.markdown("#### Settle")
    st.caption("Your call, recorded with your reasons. The input is never modified; you download a "
               "settlement export.")
    with st.form("settle"):
        options = [i.id for i in interps] + ["open"]
        chosen = st.radio("Settle on", options, index=len(options) - 1, horizontal=True)
        rationale = st.text_area("Rationale (name any straining evidence ids you have weighed)")
        submitted = st.form_submit_button("Record settlement")
    if submitted:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        settled = settle(report, chosen, rationale, stamp)
        for f in settled.findings:
            if f.stage == "settlement":
                st.markdown(f"<div class='card'>{chip(f.severity, CHIP[f.severity])}<span class='mono'>"
                            f"{f.rule_id}</span> {esc(f.message)}</div>", unsafe_allow_html=True)
        st.download_button("Download settlement export (cpr.sdl.export/1)",
                           json.dumps(to_export(settled), indent=2), file_name="settlement.json",
                           mime="application/json")
