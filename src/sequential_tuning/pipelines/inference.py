from __future__ import annotations

from pathlib import Path

from sequential_tuning.config import ProjectConfig
from sequential_tuning.providers.base import Message
from sequential_tuning.providers.factory import build_inference_provider
from sequential_tuning.utils.io import read_jsonl, write_jsonl


def run_inference(config: ProjectConfig, dataset_path: str, checkpoint_name: str, task_group: str) -> dict:
    adapter_path = None
    if checkpoint_name == "checkpoint_1":
        adapter_path = config.stage1.output_dir
    elif checkpoint_name == "checkpoint_2":
        adapter_path = config.stage2.output_dir

    provider = build_inference_provider(config, adapter_path=adapter_path)
    rows = read_jsonl(dataset_path)

    max_tokens = 96 if task_group == "alpaca" else 192
    total = min(len(rows), config.evaluation.max_prompts)

    output_path = Path(config.evaluation.output_root) / checkpoint_name / f"{task_group}_predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_outputs = []
    completed_prompt_ids = set()

    if output_path.exists():
        existing_outputs = read_jsonl(output_path)
        completed_prompt_ids = {row["prompt_id"] for row in existing_outputs}
        print(f"Resuming {checkpoint_name} {task_group}: found {len(existing_outputs)} existing predictions")

    outputs = list(existing_outputs)

    for row in rows[:total]:
        if row["prompt_id"] in completed_prompt_ids:
            continue

        if task_group == "json":
            prompt = (
                f"Instruction: {row['instruction']}\n"
                f"Input: {row['input']}\n"
                "Return only valid JSON.\n"
                "Do not include any explanation.\n"
                "Do not use markdown code fences.\n"
                "Do not include introductory text.\n"
            )
        else:
            prompt = f"Instruction: {row['instruction']}\nInput: {row['input']}"

        generated = provider.generate([Message(role="user", content=prompt)], max_tokens=max_tokens)

        outputs.append(
            {
                "prompt_id": row["prompt_id"],
                "split": row["split"],
                "task_type": row["task_type"],
                "checkpoint": checkpoint_name,
                "instruction": row["instruction"],
                "input": row["input"],
                "reference_output": row["output"],
                "prediction": generated,
                "metadata": row.get("metadata", {}),
            }
        )

        write_jsonl(outputs, output_path)

        if len(outputs) % 10 == 0:
            print(f"Generated {len(outputs)} / {total} for {checkpoint_name} {task_group}")

    return {"output_path": str(output_path), "count": len(outputs)}

