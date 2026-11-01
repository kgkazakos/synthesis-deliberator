# synthesis-deliberator

Slow synthesis: three competing interpretations of the evidence, held open until you settle.

CPR Phase II, Frame A (Test-Time Deliberation), tool 3. It reads a research question and the evidence gathered for it, and surfaces three different readings of that evidence, each tied to what it explains and what it struggles to explain, for a researcher to weigh. It never says which reading is true and never settles on one. The human is the epistemic ground.

## What it does

synthesis-deliberator reads a question, an evidence set (quotes, survey answers, analytics notes, tickets) and any interpretations you already hold, and produces:

1. Three competing interpretations from a reasoning model. Each has a statement, a mechanism, the evidence it supports and strains against, its assumptions, and one observation that would tell it apart from the others. If you supplied one or two readings, the model writes only the missing ones, and they must differ from yours. These lead the output.
2. A verifier that checks every reading, yours and the model's, with the same rules: does it cite evidence that exists, does it admit what it cannot explain, does it name a discriminating observation, is it the same story as another reading in different words, does its support rest on one participant or on synthetic data.
3. An evidence x interpretation matrix, ordered so that the evidence that tells readings apart comes first and the evidence consistent with all of them comes last.
4. Readiness checks on the evidence itself: thin, dominated by one participant, from a single source, synthetic past the threshold, or with missing and duplicate ids.
5. A settlement check. When you settle on a reading, or decide to keep all three open, you record a rationale, and the tool flags evidence that strains against your choice but goes unmentioned in your rationale.

Items 2 to 5 run with no model. With `--no-llm` the tool verifies the readings you wrote and gives you deliberation prompts for the ones you have not.

Look at the matrix's diagnostic rows first. Evidence that supports one reading and strains another is where your next question to the data, or to participants, should go.

## Three readings before settling

Synthesis usually goes wrong by converging too early. The first plausible story absorbs the data, and everything after it is read as confirmation. The method borrows from Analysis of Competing Hypotheses and from multiple working hypotheses: keep several explanations alive, test evidence against all of them at once, and pay most attention to the evidence that discriminates between them.

- **Three, exactly.** Fewer than three fires `PREMATURE_CONVERGENCE` (high). More than three is a low note: not a failure, but often a sign that two readings are variants of one.
- **Origin is always visible.** Every reading is marked `researcher` or `model`, on its card and in the export. The model never edits, restates or replaces your readings. Its readings are appended with their own ids (`M1`, `M2`, ...).
- **Verified, not graded.** The engine labels each reading `anchored` (every cited id exists and it has support) or `unanchored`, and lists any ids it invented. Any confidence, ranking or preference the model returns is dropped on parse.
- **Diagnosticity, not ranking.** Matrix rows are ordered `diagnostic` (supports one reading, strains another), `partial`, `uncited`, then `consistent` (supports every reading, so it cannot tell them apart). Readings themselves are never scored or ordered.
- **Settling is yours.** `settle` records your choice, or `open`, with a rationale and a timestamp, in a new export. It never modifies the input file and never calls a model.

## AI paradigm

LLM (reasoning / test-time deliberation).

Synthesis fails most often by converging too early: the first plausible story absorbs the data and the rest is read as confirmation. Holding three genuinely different interpretations of the same evidence open, each tied to what it explains and what it struggles to explain, is open-ended abductive reasoning, so a reasoning model spending deliberate compute on it is the right fit. A deterministic layer checks every interpretation against the evidence the researcher supplied, flags readings that are the same story reworded, and never picks a winner. Settling is the researcher's act.

Run `sdl paradigm` to print this at the tool level.

## No synthetic data, no Provenance Hash

synthesis-deliberator produces no synthetic data. It reads evidence and emits interpretations, findings and a matrix. There is no `_cpr_synthetic` block in its export, by design: model readings are analytic text, marked with origin `model` and the model id, and are not data about people.

It does consume evidence that may be synthetic. Each item's `_cpr_synthetic` ancestry is read recursively, using the same grounding factors as `provenance-inspector`. Above 30% of the whole evidence set, `SYNTHETIC_SHARE` fires; above 30% of one reading's support, `SYNTHETIC_SUPPORT` fires for that reading.

## Install

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,ui]" --no-build-isolation
```

The reasoning layer uses `litellm`, a core dependency, so any provider works. If the model is unavailable, or a call fails, the tool shows a "model unavailable" note and the report is exactly the `--no-llm` report.

### Choosing a model

The model is picked in this order: the `--model` flag (or the Streamlit model picker), then the `CPR_LLM_MODEL` environment variable, then the default `anthropic/claude-opus-5-5`. Reasoning is the product here, so the default is a strong reasoning model; `anthropic/claude-sonnet-5-5` is cheaper and faster. Each provider needs its own key:

```bash
export ANTHROPIC_API_KEY=...   # sdl run evidence.json --model anthropic/claude-opus-5-5
export OPENAI_API_KEY=...      # sdl run evidence.json --model openai/gpt-5.5
export GEMINI_API_KEY=...      # sdl run evidence.json --model gemini/gemini-3.5-flash
ollama pull llama3.1           # sdl run evidence.json --model ollama/llama3.1   (local, no key)
```

## CLI

```bash
sdl run examples/onboarding-dropoff.json                    # model writes three readings, engine verifies
sdl run examples/premature.json                             # model completes your two readings to three
sdl run examples/premature.json --no-llm                    # verifier and prompts only, no model
sdl run examples/thin-evidence.json --min-evidence 4        # adjust a threshold
sdl export examples/onboarding-dropoff.json -o out.json     # versioned JSON (cpr.sdl.export/1)
sdl matrix out.json                                         # evidence x interpretation matrix only
sdl settle out.json --choose M2 --rationale "..." -o s.json # record your settlement in a new export
sdl paradigm                                                # paradigm + rationale
```

`run` and `export` call a model and accept `--model` and `--no-llm`. The model is called only when fewer than three readings are supplied. `matrix`, `settle` and `paradigm` never call a model. `run`, `matrix` and `settle` accept an input file or an export; settle on an export to settle on model readings.

Thresholds, on `run`, `export`, `matrix` and `settle`:

| Option | Default | Rule |
|--------|---------|------|
| `--min-evidence` | 6 | `THIN_EVIDENCE` fires below this many items |
| `--dominance` | 0.5 | `PARTICIPANT_DOMINANCE` fires when one participant's share of attributed items is above this |
| `--overlap-support` | 0.7 | `NEAR_DUPLICATE` fires when two readings' supporting ids overlap (Jaccard) above this |
| `--overlap-statement` | 0.6 | `NEAR_DUPLICATE` fires when two statements' content words overlap (Jaccard) above this |

## Streamlit

```bash
streamlit run synthesis_deliberator/app.py
```

Frame A visual language: deep violet accent `#4f3d8f`, DM Sans and DM Mono, near-black sidebar, warm-beige controls, cream main, card rows with severity chips. The sidebar holds a model picker with Anthropic, OpenAI, Google and local Ollama presets, plus a free-text field for any litellm id; the controls column holds the input and the four thresholds. The three readings render side by side as cards with origin, anchoring and flag chips, model readings labelled as additive with the model id, then the matrix, the readiness findings, and a settle form that downloads a settlement export.

## The verifier

The engine is a set of structural checks over evidence ids, participants, sources and wording, honest about being one. It is fast, reproducible and cheap, and it is not an oracle: it cannot tell which reading is right, only whether each one is anchored, distinct and candid about what it leaves unexplained. Output depends only on the input and the thresholds.

| Rule | Stage | Fires when | Severity |
|------|-------|-----------|----------|
| `THIN_EVIDENCE` | readiness | Fewer than `--min-evidence` items | medium |
| `MISSING_ID` | readiness | An evidence item has no id (shown as `#n`) | high |
| `DUPLICATE_ID` | readiness | Two items share an id | high |
| `PARTICIPANT_DOMINANCE` | readiness | One participant's share is above `--dominance` | medium |
| `SINGLE_SOURCE` | readiness | All attributed evidence comes from one source | medium |
| `SYNTHETIC_SHARE` | readiness | Synthetic share of the evidence above 30% | high |
| `PREMATURE_CONVERGENCE` | verifier | Fewer than three readings | high |
| `EXTRA_INTERPRETATIONS` | verifier | More than three readings | low |
| `UNANCHORED` | verifier | A reading cites an id that does not exist, or cites no support | high |
| `DOUBLE_CITED` | verifier | A reading cites one item as both support and strain | low |
| `NO_STRAIN` | verifier | A reading names no evidence it struggles to explain | medium |
| `NO_DISCRIMINATOR` | verifier | A reading names no observation that would tell it apart | medium |
| `NEAR_DUPLICATE` | verifier | Two readings overlap above either overlap threshold | medium |
| `SINGLE_VOICE_SUPPORT` | verifier | All of a reading's support comes from one participant | medium |
| `SYNTHETIC_SUPPORT` | verifier | Synthetic share of a reading's support above 30% | high |
| `ORPHANED_EVIDENCE` | verifier | Evidence cited by no reading | low |
| `NO_RATIONALE` | settlement | A settlement has no rationale | high |
| `UNADDRESSED_STRAIN` | settlement | The chosen reading strains against ids the rationale never mentions | medium |

## Input format

A JSON object with `question`, `evidence` and, optionally, `interpretations`. Evidence items take `id`, `text`, `participant` and `source` (or `method`); items without an id get a positional `#n`. Interpretations in an input file are always the researcher's; missing ids default to `I1`, `I2` and so on.

```json
{
  "question": "Why do first-time workspace admins abandon setup?",
  "evidence": [
    {"id": "E1", "participant": "P1", "source": "interview",
     "text": "I got to the roles screen and didn't know who should be an owner."},
    {"id": "E7", "source": "analytics",
     "text": "62% of abandoned setups stop on the roles and permissions step."}
  ],
  "interpretations": [
    {"id": "I1",
     "statement": "Admins stall because the roles model is hard to map onto their organisation.",
     "mechanism": "The role model forces an org-design decision the admin did not expect.",
     "supports": ["E1", "E7"], "strains": ["E6"],
     "assumptions": ["The roles step is where people stop, not merely pause."],
     "discriminator": "Abandonment drops when the roles step is deferred."}
  ]
}
```

An evidence item may carry a `_cpr_synthetic` block from an upstream CPR tool; its ancestry feeds `SYNTHETIC_SHARE` and `SYNTHETIC_SUPPORT`.

See `examples/` for:

- `onboarding-dropoff.json`: ten items from interviews, a survey, analytics and support, with no readings. The model writes all three; under `--no-llm` it fires `PREMATURE_CONVERGENCE` and prints prompts.
- `premature.json`: the same evidence with two researcher readings that are one story (`PREMATURE_CONVERGENCE`, `NEAR_DUPLICATE`, `UNANCHORED`, `NO_STRAIN`, `NO_DISCRIMINATOR`, `DOUBLE_CITED`, `ORPHANED_EVIDENCE`).
- `thin-evidence.json`: five interview items, mostly from one participant, partly synthetic, with a missing and a duplicated id, and four readings (every readiness rule, `SYNTHETIC_SUPPORT`, `SINGLE_VOICE_SUPPORT`, `NEAR_DUPLICATE`, `EXTRA_INTERPRETATIONS`).
- `three-readings.json`: three distinct, anchored readings of the onboarding evidence, which fires nothing.
- `three-readings.settled.json`: an export made by `sdl settle` that settles on `I2` with no rationale (`NO_RATIONALE`, `UNADDRESSED_STRAIN`).

## Export schema

`cpr.sdl.export/1`: `schema`, `exported_at`, `paradigm` and `report`. The report holds `question`, `thresholds`, `evidence` (with each item's computed `syntheticity`), `interpretations` (each with `origin`), `checks` (anchoring status, unknown ids, participant count and synthetic share per reading), `matrix` (marks per reading and `diagnosticity`), `findings` (with `stage`), `prompts`, `synthetic_share`, `settlement`, `model` (the id that wrote any model readings) and `model_note`. There is no score or ranking field.

## Build

- Model-first: a reasoning model's interpretations lead, a rule-based verifier checks every reading with the same rules, and the tool falls back to the `--no-llm` report when the model is unavailable.
- `paradigm` command, and `--no-llm` for the deterministic checks alone.
- pytest suite, CI on Python 3.10, 3.11, 3.12.
- Versioned export `cpr.sdl.export/1`.

## IP

A single-model synthesis tool: no synthetic data produced, no agents, no orchestration, no automatic linkage to external systems. One model call writes hypotheses about the evidence; the engine checks them against that evidence, never against each other's arguments, and the researcher settles.

## License

MIT.
