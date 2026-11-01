# cpr-tool

TEMPLATE: one-line tagline. What the tool is, in one sentence.

TEMPLATE: CPR Phase II, Frame X (Frame Name), tool N. One or two sentences on what it reads and what it surfaces for a researcher to weigh. What it never does (for example, it never issues a verdict). The human is the epistemic ground.

## What it does

TEMPLATE: what the tool reads, then a numbered list of what it produces:

1. The deterministic output of the rule-based core. This runs with no model.
2. What the AI layer adds, and that it is additive.

TEMPLATE: one short paragraph on what the researcher should look at first, and why.

## TEMPLATE: signature concept

TEMPLATE: the one idea or mode that makes this tool distinctive (for example
"Verification-chain mode" or "How divergence is scored"). Use a short list or table where
it helps. Delete this section if the tool has no such concept.

## AI paradigm

TEMPLATE: paradigm name.

TEMPLATE: the rationale paragraph, identical to `AI_PARADIGM_RATIONALE`: why this paradigm
fits this research task.

Run `cpr-tool paradigm` to print this at the tool level.

## No synthetic data, no Provenance Hash

TEMPLATE: if the tool produces no synthetic data, keep this heading and say so: what it
reads, what it emits, no `_cpr_synthetic` block by design, and why provenance theatre would
be noise. If it does produce synthetic data, rename the heading to "Synthetic Provenance
Hash" and describe the block it emits, its `human_grounding` value and the ancestry it reads.

## Install

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,ui]" --no-build-isolation
```

TEMPLATE: one or two sentences on the AI layer: it uses `litellm` (a core dependency), set a
backend with `CPR_LLM_MODEL` (default `gpt-4o-mini`) and the usual provider key, and with no
backend the tool degrades gracefully to the deterministic core.

## CLI

```bash
cpr-tool run examples/sample.json             # deterministic core + AI layer
cpr-tool run examples/sample.json --no-llm    # deterministic core only
cpr-tool export examples/sample.json -o out.json  # versioned JSON (cpr.tool.export/1)
cpr-tool paradigm                             # paradigm + rationale
```

Every command that can call a model accepts `--no-llm` and `--model`.

## Streamlit

```bash
streamlit run cpr_tool/app.py
```

TEMPLATE: Frame X visual language: accent name `#hex`, DM Sans and DM Mono, three-column
layout, cream main, card rows with severity chips. One sentence on how the main output renders.

## TEMPLATE: the engine

TEMPLATE: heading named for the deterministic core (for example "The rule library").
One paragraph on what the core is and is not, honestly, then its rules or algorithm as a
list or table.

## Input format

TEMPLATE: the input shape, which fields are required, and a short JSON example.

```json
{
  "title": "Example"
}
```

See `examples/` for TEMPLATE: what each example demonstrates.

## Build

- Rule-based deterministic core, additive AI layer, graceful degradation.
- `paradigm` command and `--no-llm` flag, per the Phase II conventions.
- pytest suite, CI on Python 3.10, 3.11, 3.12.
- Versioned export `cpr.tool.export/1`.

## IP

TEMPLATE: one or two neutral sentences, for example "Clean. A pure critique tool: no
synthetic data, no agents, no orchestration, no causal auto-linkage. Not on any constraint
list." Never name employers, filings or private projects here.

## License

MIT.
