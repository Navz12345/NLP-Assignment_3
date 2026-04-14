from __future__ import annotations

import json
from pathlib import Path

from sequential_tuning.config import ProjectConfig
from sequential_tuning.providers.base import Message
from sequential_tuning.providers.factory import build_teacher_provider
from sequential_tuning.utils.io import read_jsonl, write_json, write_jsonl
from sequential_tuning.utils.json_eval import parse_json_safe
from sequential_tuning.utils.templates import load_template


def generate_teacher_dataset(config: ProjectConfig) -> dict:
    provider = build_teacher_provider(config)
    prompts_dir = config.runtime.prompts_dir
    system_prompt = load_template(prompts_dir, "teacher_system_prompt.txt")
    train_seed_path = Path(config.data.teacher_train_path).with_name("teacher_seed_train.jsonl")
    eval_seed_path = Path(config.data.teacher_eval_path).with_name("teacher_seed_eval.jsonl")
    train_rows = []
    invalid_rows = []
    for row in read_jsonl(train_seed_path):
        user_prompt = (
            f"Task Type: {row['task_type']}\n"
            f"Instruction: {row['instruction']}\n"
            f"Input: {row['input']}\n"
            f"Schema: {json.dumps(row['metadata'].get('schema', {}))}"
        )
        response = provider.generate(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            max_tokens=512,
        )
        is_valid, parsed, error = parse_json_safe(response)
        if not is_valid:
            invalid_rows.append({"prompt_id": row["prompt_id"], "error": error, "raw_output": response})
            continue
        train_rows.append(
            {
                **row,
                "output": json.dumps(parsed, ensure_ascii=False),
                "metadata": {
                    **row.get("metadata", {}),
                    "teacher_model": config.models.teacher_model_name,
                    "json_validated": True,
                },
            }
        )
    eval_rows = read_jsonl(eval_seed_path)
    write_jsonl(train_rows, config.data.teacher_train_path)
    write_jsonl(eval_rows, config.data.teacher_eval_path)
    report = {
        "train_generated_count": len(train_rows),
        "eval_count": len(eval_rows),
        "invalid_count": len(invalid_rows),
        "teacher_model": config.models.teacher_model_name,
    }
    write_json(report, Path(config.runtime.artifacts_root) / "manifests" / "teacher_generation_report.json")
    if invalid_rows:
        write_json(invalid_rows, Path(config.runtime.artifacts_root) / "manifests" / "teacher_invalid_rows.json")
    return report

