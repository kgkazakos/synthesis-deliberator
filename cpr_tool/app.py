"""Streamlit UI. Run: streamlit run cpr_tool/app.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script, so relative imports break. Use the repo root + absolute import.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cpr_tool import __version__  # noqa: E402  TEMPLATE: rename package
from cpr_tool.engine import analyze  # noqa: E402
from cpr_tool.models import AI_PARADIGM, AI_PARADIGM_RATIONALE, TOOL_NAME  # noqa: E402

ACCENT = "#4f3d8f"  # TEMPLATE: frame accent, keep in sync with .streamlit/config.toml
CHIP = {"high": "#9b2c2c", "medium": "#b7791f", "low": "#4a5568"}

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
.card {{ background: #fff; border: 1px solid #e8e2d6; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }}
.chip {{ display: inline-block; padding: 1px 8px; border-radius: 999px; color: #fff; font-size: 12px; margin-right: 8px; }}
.accent {{ color: {ACCENT}; }}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"### {TOOL_NAME}")
    st.caption(f"v{__version__} · {AI_PARADIGM}")
    with st.expander("Why this paradigm"):
        st.write(AI_PARADIGM_RATIONALE)

controls, main = st.columns([1, 2.4])
with controls:
    st.markdown("#### Input")
    raw = st.text_area("JSON object", value='{"title": "", "goal": "Example"}', height=200)

with main:
    st.markdown("<h3 class='accent'>Findings</h3>", unsafe_allow_html=True)
    try:
        doc = json.loads(raw)
        report = analyze(doc if isinstance(doc, dict) else {"value": doc})
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        st.stop()
    if not report.findings:
        st.info("No findings.")
    for f in report.findings:
        st.markdown(
            f"<div class='card'><span class='chip' style='background:{CHIP[f.severity]}'>{f.severity}</span>"
            f"<span class='mono'>{f.rule_id}</span><br/>{f.message}</div>",
            unsafe_allow_html=True,
        )
