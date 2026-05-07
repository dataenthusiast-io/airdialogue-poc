# Zero-Shot Intent Classification PoC

A reproducible pipeline evaluating GPT-4o-mini as a zero-shot intent classifier for airline customer service conversations, using the [Google AirDialogue](https://huggingface.co/datasets/google/air_dialogue) dataset.

---

## Research Questions

**RQ1:** Can GPT-4o-mini reliably classify airline customer service intent in a zero-shot setting, without any labeled training examples?

**RQ2:** What are the cost and schema reliability characteristics of LLM-based classification at industrial scale?

| | Hypothesis | Threshold |
|---|---|---|
| **H1** | Zero-shot classification is viable | Macro F1 ≥ 0.80 |
| **H2** | Structured output is reliable at scale | Schema failure rate < 5% |

---

## Results

| Metric | Value |
|---|---|
| Macro F1 | **1.000** |
| Schema failure rate | **0.0%** |
| Cost per call | $0.000083 |
| Projected cost (10K calls) | $0.83 |

Both H1 and H2 supported. Results are in `results/evaluation_report.json`.

---

## Dataset

[Google AirDialogue](https://huggingface.co/datasets/google/air_dialogue) (`air_dialogue_data`, train split) — 402,038 goal-oriented airline customer service dialogues with ground-truth intent labels (`book`, `change`, `cancel`). A stratified sample of 300 conversations (100 per class) is used for evaluation.

---

## Project Structure

```
airdialogue-poc/
├── config/
│   └── config.yaml          # All runtime constants
├── data/
│   ├── raw/                  # HF dataset cache (gitignored)
│   └── processed/
│       ├── sample.jsonl      # 300-row eval sample (no labels)
│       └── ground_truth.jsonl
├── prompts/
│   └── system.yaml          # Zero-shot system prompt
├── results/
│   ├── results.jsonl        # Per-call LLM output
│   ├── evaluation_report.json
│   └── plots/               # 4 PNG figures
├── src/
│   ├── config.py            # Typed config loader
│   ├── data_pipeline.py     # Phase 1
│   ├── schemas.py           # Pydantic output schema
│   ├── prompts.py           # Phase 2
│   ├── pipeline.py          # Phase 3
│   ├── evaluate.py          # Phase 4
│   └── visualize.py         # Phase 4
├── run_pipeline.py          # Phase 3 entrypoint
└── run_evaluate.py          # Phase 4 entrypoint
```

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

---

## Running the Pipeline

**Phase 1 — Build evaluation sample** (requires HF dataset download, ~500 MB):

```bash
python3 -m src.data_pipeline
```

Writes `data/processed/sample.jsonl` (dialogues only) and `data/processed/ground_truth.jsonl` (labels only, withheld until Phase 4).

**Phase 3 — Run LLM classification** (~300 API calls, ~$0.025):

```bash
python3 run_pipeline.py
```

Writes `results/results.jsonl` with one row per call. Supports resume — re-running skips already-processed IDs.

**Phase 4 — Evaluate and generate plots:**

```bash
python3 run_evaluate.py
```

Writes `results/evaluation_report.json` and 4 PNG plots to `results/plots/`.

---

## Output

### `results/evaluation_report.json`

```json
{
  "classification": {
    "macro_f1": 1.0,
    "per_class": {
      "book":   {"precision": 1.0, "recall": 1.0, "f1": 1.0},
      "change": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
      "cancel": {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    },
    "confusion_matrix": { "..." : "..." }
  },
  "operational": {
    "total_calls": 300,
    "schema_failures": 0,
    "schema_failure_rate": 0.0,
    "mean_cost_usd": 0.000083,
    "projected_cost_usd_10k": 0.83
  },
  "hypothesis": {
    "h1_supported": true,
    "h2_supported": true
  }
}
```

### Plots (`results/plots/`)

| File | Description |
|---|---|
| `confusion_matrix.png` | Row-normalised heatmap |
| `f1_per_class.png` | Per-class F1 bar chart with H1 threshold line |
| `token_distribution.png` | Prompt token count distribution |
| `cost_projection.png` | Cumulative cost up to 10,000 interactions |

---

## Configuration

All runtime constants are in `config/config.yaml` — no hardcoded values elsewhere. Key settings:

```yaml
model:
  name: gpt-4o-mini
  temperature: 0

sampling:
  total: 300
  per_class: 100
  random_seed: 42

costs:
  input_per_1k_tokens: 0.000150
  output_per_1k_tokens: 0.000600
  industrial_scale_benchmark: 10000
```

---

## Reproducibility

- Fixed `random_seed: 42` — same sample every run
- Temperature 0 — deterministic LLM outputs
- Resume logic — interrupted runs continue from last completed ID
- Phase 4 is read-only — results and processed data are never modified after their phase completes
