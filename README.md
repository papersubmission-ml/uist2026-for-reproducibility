# uist2026-for-reproducibility

Reproducible workflow for the `uist2026-for-reproducibility` package.

## Tree

```text
uist2026-for-reproducibility/
├── README.md
├── data/HAI-UIST-DATA/
├── webapp/
├── shared/json_export_loader.py
├── benchmark/
│   ├── llm_judge_augment_json.py
│   ├── visualize_results.py
│   └── outputs/
└── analysis/
    ├── counterfactual_study_analysis.py
    ├── counterfactual_study_plots.py
    ├── user_rating_analysis.py
    ├── final_plan_quality_analysis.py
    ├── plan_source_analysis.py
    ├── UIST_evaluation_metrics.md
    └── outputs/
```

## Inputs

`webapp/` stores the study website used to collect participant data and run the
user-facing study flow.

Required raw files:

- `data/HAI-UIST-DATA/all-completed-HAI-2026-03-27.json`
- `data/HAI-UIST-DATA/all-completed-AIH-2026-03-27.json`
  
We are currently in the process of anonymizing. We will be releasing the processed data once the paper is accepted.

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
- `sentence-transformers`: also needed for `analysis/counterfactual_study_analysis.py`
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

## Benchmark

LLM-as-a-Judge JSON augmentation:

Creates augmented copies of the input JSON files under
`benchmark/outputs/llm_judge_augmented_outputs/<timestamp>/augmented_json/`.
For each session and for each AI suggestion (`LLM-B`, `LLM-CF`, `LLM-CT`), it
adds a new `LLM_as_a_Judge` field inside `AI_Suggestions[<model>]`. That field
contains four 1-5 ratings:

- `Action_Clarity`
- `Feasibility`
- `Goal_Alignment`
- `Insight_Novelty`

The original input JSON files are not modified.

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

Counterfactual study cleaned CSV builder:

Builds the legacy per-model CSV for the counterfactual analysis notebook from
the JSON exports. It includes participant ratings, selected final-plan fields,
plan-source similarity matrices, and `llm_judge_*` scores.

If `LLM_as_a_Judge` is not in the raw JSON, the script automatically loads the
matching augmented JSON under `benchmark/outputs/llm_judge_augmented_outputs/`.

```bash
python analysis/counterfactual_study_analysis.py \
  --inputs data/HAI-UIST-DATA/all-completed-HAI-2026-03-27.json \
           data/HAI-UIST-DATA/all-completed-AIH-2026-03-27.json \
  --output-csv analysis/outputs/cleaned_data/df_all_cleaned_with_llm_judge_with_plan_source_vectors.csv
```

Counterfactual study plots:

Builds the notebook-style tables and Plotly figures from the cleaned CSV.

```bash
python analysis/counterfactual_study_plots.py \
  --input-csv analysis/outputs/cleaned_data/df_all_cleaned_with_llm_judge_with_plan_source_vectors.csv \
  --output analysis/outputs/counterfactual_study/main
```

## Notes

- Run all commands from `uist2026-for-reproducibility/`.
- Main outputs are written to `benchmark/outputs/` and `analysis/outputs/`.
