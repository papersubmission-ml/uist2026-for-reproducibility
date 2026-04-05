# uist2026-for-reproducibility

Reproducible workflow for the `uist2026-for-reproducibility` package.

## Tree

```text
uist2026-for-reproducibility/
├── README.md
├── data/HAI-UIST-DATA/
├── shared/json_export_loader.py
├── benchmark/
│   ├── llm_judge_augment_json.py
│   ├── visualize_results.py
│   └── outputs/
└── analysis/
    ├── user_rating_analysis.py
    ├── final_plan_quality_analysis.py
    ├── plan_source_analysis.py
    ├── feedback_llm_judge.py
    ├── plotly_ai_suggestion_analysis.py
    ├── UIST_evaluation_metrics.md
    └── outputs/
```

## Inputs

Required raw files:

- `data/HAI-UIST-DATA/all-completed-HAI-2026-03-27.json`
- `data/HAI-UIST-DATA/all-completed-AIH-2026-03-27.json`

## Setup

Use Python 3.10 or higher. Python 3.11 is recommended. Python 3.9 will not
work because several scripts use Python 3.10 type-union syntax.

`venv` is recommended:

```bash
cd uist2026-for-reproducibility
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "numpy<2" pandas matplotlib scipy scikit-learn \
  plotly packaging tqdm pydantic openai torch "transformers<4.57"
```

Optional:

```bash
python -m pip install sentence-transformers "kaleido<1"
```

- `sentence-transformers`: needed for `analysis/plan_source_analysis.py`
- `kaleido<1`: needed only for Plotly PNG export
- If you already installed newer packages, repair the benchmark environment with:
  `python -m pip install --upgrade "numpy<2" "transformers<4.57"`

## Environment Variable

Set this only for OpenAI-based steps:

```bash
export OPENAI_API_KEY="sk-..."
```

Used by:

- `benchmark/llm_judge_augment_json.py`
- `analysis/feedback_llm_judge.py`

## Benchmark

LLM-as-a-Judge JSON augmentation:

Adds `LLM_as_a_Judge` ratings to copied JSON files for downstream analysis.

```bash
python benchmark/llm_judge_augment_json.py \
  --inputs data/HAI-UIST-DATA/all-completed-HAI-2026-03-27.json \
           data/HAI-UIST-DATA/all-completed-AIH-2026-03-27.json \
  --model gpt-5-nano \
  --reasoning-effort low \
  --timestamp full_llm_judge_gpt5nano_low
```

## Analysis

Participant ratings:

Summarizes participant ratings for `LLM-B`, `LLM-CF`, and `LLM-CT`.

```bash
python analysis/user_rating_analysis.py \
  --input data/HAI-UIST-DATA \
  --condition HAI \
  --output analysis/outputs/user_rating_outputs/HAI
python analysis/user_rating_analysis.py \
  --input data/HAI-UIST-DATA \
  --condition AIH \
  --output analysis/outputs/user_rating_outputs/AIH
```

Final-plan quality:

Measures how final plans changed and how close selected plans are to the AI suggestions.

```bash
python analysis/final_plan_quality_analysis.py \
  --input data/HAI-UIST-DATA \
  --condition HAI \
  --output analysis/outputs/final_plan_outputs/HAI
python analysis/final_plan_quality_analysis.py \
  --input data/HAI-UIST-DATA \
  --condition AIH \
  --output analysis/outputs/final_plan_outputs/AIH
```

Plan source attribution:

Estimates whether final plans came from the initial plans, the AI suggestions, or a new source.

```bash
python analysis/plan_source_analysis.py \
  --input data/HAI-UIST-DATA \
  --output analysis/outputs/plan_source_outputs/main
```

Feedback coding:

Codes free-text participant feedback with an OpenAI rubric.

```bash
python analysis/feedback_llm_judge.py \
  --input data/HAI-UIST-DATA \
  --condition HAI \
  --output analysis/outputs/feedback_outputs/HAI
python analysis/feedback_llm_judge.py \
  --input data/HAI-UIST-DATA \
  --condition AIH \
  --output analysis/outputs/feedback_outputs/AIH
```

Plotly figures from human ratings:

Builds interactive HTML figures from the original participant rating data.

```bash
python analysis/plotly_ai_suggestion_analysis.py \
  --inputs data/HAI-UIST-DATA/all-completed-HAI-2026-03-27.json \
           data/HAI-UIST-DATA/all-completed-AIH-2026-03-27.json \
  --timestamp human_plotly \
  --html-only
```

Plotly figures from LLM-as-a-Judge outputs:

Builds the same interactive figures using `LLM_as_a_Judge` ratings.

```bash
python analysis/plotly_ai_suggestion_analysis.py \
  --inputs benchmark/outputs/llm_judge_augmented_outputs/full_llm_judge_gpt5nano_low/augmented_json/all-completed-HAI-2026-03-27.json \
           benchmark/outputs/llm_judge_augmented_outputs/full_llm_judge_gpt5nano_low/augmented_json/all-completed-AIH-2026-03-27.json \
  --rating-source llm_judge \
  --timestamp llm_judge_plotly \
  --html-only
```

## Notes

- Run all commands from `uist2026-for-reproducibility/`.
- Main outputs are written to `benchmark/outputs/` and `analysis/outputs/`.
