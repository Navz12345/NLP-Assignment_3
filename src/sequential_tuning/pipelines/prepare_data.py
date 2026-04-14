from __future__ import annotations

import random
from pathlib import Path

from sequential_tuning.config import ProjectConfig
from sequential_tuning.utils.io import read_json, read_jsonl, write_json, write_jsonl


def _normalize_record(record: dict, prompt_id: str, split: str, task_type: str) -> dict:
    return {
        "prompt_id": prompt_id,
        "split": split,
        "task_type": task_type,
        "instruction": record.get("instruction", "").strip(),
        "input": record.get("input", "").strip(),
        "output": record.get("output", "").strip(),
        "metadata": record.get("metadata", {}),
    }


def prepare_alpaca_data(config: ProjectConfig) -> dict:
    source_path = Path(config.data.alpaca_raw_path)
    if source_path.suffix == ".jsonl":
        raw_rows = read_jsonl(source_path)
    else:
        raw_rows = read_json(source_path)
    cleaned_rows = []
    for idx, row in enumerate(raw_rows):
        if not row.get("instruction") or not row.get("output"):
            continue
        cleaned_rows.append(_normalize_record(row, f"alpaca_{idx:04d}", "train", row.get("task_type", "general_instruction")))
    random.Random(config.data.random_seed).shuffle(cleaned_rows)
    eval_size = min(max(1, config.data.sample_size), max(1, len(cleaned_rows) // 5))
    eval_rows = [dict(row, split="eval") for row in cleaned_rows[:eval_size]]
    train_rows = [dict(row, split="train") for row in cleaned_rows[eval_size:]]
    write_jsonl(train_rows, config.data.alpaca_train_path)
    write_jsonl(eval_rows, config.data.alpaca_eval_path)
    manifest = {
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "source_path": str(source_path),
    }
    write_json(manifest, Path(config.runtime.artifacts_root) / "manifests" / "alpaca_prepare.json")
    return manifest


def prepare_teacher_seed_prompts(config: ProjectConfig) -> dict:
    prompt_rows = read_json(config.data.teacher_prompt_seed_path)
    train_rows = []
    eval_rows = []
    for idx, row in enumerate(prompt_rows):
        normalized = _normalize_record(
            {
                "instruction": row["instruction"],
                "input": row.get("input", ""),
                "output": row.get("output", ""),
                "metadata": {
                    "schema": row.get("schema", {}),
                    "task_type": row.get("task_type", "json_task"),
                    "reference_output": row.get("reference_output"),
                },
            },
            prompt_id=f"json_{idx:04d}",
            split=row.get("split", "train"),
            task_type=row.get("task_type", "json_task"),
        )
        if normalized["split"] == "eval":
            eval_rows.append(normalized)
        else:
            train_rows.append(normalized)
    write_jsonl(train_rows, Path(config.data.teacher_train_path).with_name("teacher_seed_train.jsonl"))
    write_jsonl(eval_rows, Path(config.data.teacher_eval_path).with_name("teacher_seed_eval.jsonl"))
    manifest = {
        "train_seed_count": len(train_rows),
        "eval_seed_count": len(eval_rows),
        "source_path": config.data.teacher_prompt_seed_path,
    }
    write_json(manifest, Path(config.runtime.artifacts_root) / "manifests" / "teacher_seed_prepare.json")
    return manifest

