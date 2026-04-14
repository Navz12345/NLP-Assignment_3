# Sequential Instruction Tuning for Assignment 3

This repository is structured around the exact pipeline required in the assignment PDF:

1. Prepare Alpaca-style instruction data
2. Construct a teacher-generated JSON instruction dataset
3. Prepare Stage 1 and Stage 2 QLoRA training jobs
4. Run inference at checkpoints 0, 1, and 2
5. Evaluate with automatic metrics and an LLM judge
6. Aggregate results for the final report and forgetting analysis

## Repository layout

- `configs/`: sample and full experiment configs
- `prompts/`: editable teacher and judge prompts
- `data/`: raw, interim, processed, and sample datasets
- `src/sequential_tuning/`: modular pipeline code
- `hpc/`: UTSA HPC batch scripts for Stage 1 and Stage 2
- `artifacts/`: generated manifests, checkpoints, logs, and evaluations
- `docs/REPORT_TEMPLATE.md`: blog-post/report starter

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

For real teacher and judge calls, set environment variables instead of putting secrets into tracked files:

```bash
copy .env.example .env
```

## Quick local smoke run

This uses `configs/sample.yaml` with mock teacher, judge, and inference backends, so you can verify the pipeline wiring before any expensive training:

```bash
python -m sequential_tuning.cli --config configs/sample.yaml sanity-check
python -m sequential_tuning.cli --config configs/sample.yaml quickstart
```

That command will:

- prepare sample Alpaca and JSON datasets
- generate validated mock teacher outputs
- materialize Stage 1 and Stage 2 training plans
- produce checkpoint 0, 1, and 2 mock predictions
- compute automatic metrics
- run pairwise mock judge comparisons
- write an aggregated summary table

## Recommended workflow

Primary config:

```bash
configs/full_phi35_utsa.yaml
```

Fallback config if local Phi access/download becomes unreliable:

```bash
configs/fallback_qwen3_utsa.yaml
```

Use your shell or a local `.env` loader to export:

```bash
HF_TOKEN=...
UTSA_QWEN_API_KEY=...
UTSA_QWEN_BASE_URL=...
UTSA_LLAMA_API_KEY=...
UTSA_LLAMA_BASE_URL=...
```

The Llama endpoint is behind the UTSA VPN, so connect to VPN before teacher generation or judge runs that depend on it.

## Recommended full workflow

1. Put your Alpaca dataset in `data/raw/alpaca_cleaned.json`
2. Generate a full JSON seed set with at least 100 prompts total
3. Run `prepare-data` and `generate-teacher-data`
4. Run `sanity-check --full`
5. Submit Stage 1 and Stage 2 jobs on UTSA HPC
6. Run inference and evaluation for checkpoints 0, 1, and 2
7. Aggregate metrics and fill `docs/REPORT_TEMPLATE.md`

Suggested commands:

```bash
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml build-json-seeds --output data/raw/json_prompt_seeds_full.json --prompts-per-task 24
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml prepare-data
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml generate-teacher-data
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml prepare-training --stage all
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml sanity-check --full
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml infer --checkpoint checkpoint_0
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml infer --checkpoint checkpoint_1
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml infer --checkpoint checkpoint_2
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml evaluate --checkpoint checkpoint_0
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml evaluate --checkpoint checkpoint_1
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml evaluate --checkpoint checkpoint_2
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml judge --task-group alpaca --checkpoint-a checkpoint_0 --checkpoint-b checkpoint_1
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml judge --task-group alpaca --checkpoint-a checkpoint_1 --checkpoint-b checkpoint_2
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml judge --task-group json --checkpoint-a checkpoint_0 --checkpoint-b checkpoint_2
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml aggregate
```

If you want a guarded one-shot orchestration after your smoke tests pass:

```bash
python -m sequential_tuning.cli --config configs/full_phi35_utsa.yaml full-run
```

That command writes `artifacts/full_run_completed.flag` and refuses to rerun unless you pass `--force`.

## Notes on best practice

- Keep prompts editable in `prompts/`, not inline in scripts
- Keep held-out eval sets separate from training data
- Save all outputs by checkpoint for reproducibility
- Use the local smoke run before a single expensive full run
- Track your ablations in separate config copies or artifact subdirectories
- Never commit API keys or tokens into YAML, Python files, notebooks, or the repo history
