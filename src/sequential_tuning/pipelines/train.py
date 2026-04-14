from __future__ import annotations

import json
from pathlib import Path

from sequential_tuning.config import ProjectConfig, TrainingStageConfig
from sequential_tuning.utils.io import ensure_parent, write_json


def write_training_plan(config: ProjectConfig, stage: TrainingStageConfig, base_checkpoint: str | None = None) -> dict:
    output_dir = Path(stage.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "student_model_name": config.models.student_model_name,
        "stage_name": stage.name,
        "dataset_path": stage.dataset_path,
        "output_dir": stage.output_dir,
        "base_checkpoint": base_checkpoint,
        "epochs": stage.epochs,
        "learning_rate": stage.learning_rate,
        "per_device_train_batch_size": stage.per_device_train_batch_size,
        "gradient_accumulation_steps": stage.gradient_accumulation_steps,
        "max_seq_length": stage.max_seq_length,
        "lora": {
            "r": stage.lora_r,
            "alpha": stage.lora_alpha,
            "dropout": stage.lora_dropout,
        },
        "notes": "Use scripts/train_qlora.py or the equivalent HF Trainer entrypoint on UTSA HPC.",
    }
    write_json(plan, output_dir / "training_plan.json")
    return plan


def write_training_command(stage: TrainingStageConfig, config: ProjectConfig, base_checkpoint: str | None = None) -> str:
    base_arg = f" --resume_from_checkpoint {base_checkpoint}" if base_checkpoint else ""
    return (
        "python -m sequential_tuning.training_runner"
        f" --model_name {config.models.student_model_name}"
        f" --dataset_path {stage.dataset_path}"
        f" --output_dir {stage.output_dir}"
        f" --epochs {stage.epochs}"
        f" --learning_rate {stage.learning_rate}"
        f" --batch_size {stage.per_device_train_batch_size}"
        f" --grad_accum {stage.gradient_accumulation_steps}"
        f" --max_seq_length {stage.max_seq_length}"
        f" --lora_r {stage.lora_r}"
        f" --lora_alpha {stage.lora_alpha}"
        f" --lora_dropout {stage.lora_dropout}"
        f" --hf_token_env {config.models.student_hf_token_env}"
        f"{base_arg}"
    ).strip()


def materialize_stage_artifacts(config: ProjectConfig, stage: TrainingStageConfig, base_checkpoint: str | None = None) -> dict:
    plan = write_training_plan(config, stage, base_checkpoint=base_checkpoint)
    command = write_training_command(stage, config, base_checkpoint=base_checkpoint)
    command_path = ensure_parent(Path(stage.output_dir) / "launch_command.txt")
    command_path.write_text(command + "\n", encoding="utf-8")
    return {"plan": plan, "command": command}


def create_placeholder_adapter(stage: TrainingStageConfig) -> None:
    adapter_path = ensure_parent(Path(stage.output_dir) / "adapter_config.json")
    adapter_path.write_text(json.dumps({"status": "placeholder", "stage": stage.name}, indent=2), encoding="utf-8")

