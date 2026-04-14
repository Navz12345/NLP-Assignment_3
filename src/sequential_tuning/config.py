from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    student_model_name: str
    teacher_provider: str
    teacher_model_name: str
    judge_provider: str
    judge_model_name: str
    inference_provider: str = "mock"
    inference_model_name: str = "mock-student"
    student_hf_token_env: str = "HF_TOKEN"
    teacher_api_key_env: str = "OPENAI_API_KEY"
    teacher_base_url_env: str = "OPENAI_BASE_URL"
    judge_api_key_env: str = "OPENAI_API_KEY"
    judge_base_url_env: str = "OPENAI_BASE_URL"
    inference_api_key_env: str = "OPENAI_API_KEY"
    inference_base_url_env: str = "OPENAI_BASE_URL"


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_values(item) for key, item in value.items()}
    return value


@dataclass
class TrainingStageConfig:
    name: str
    dataset_path: str
    output_dir: str
    epochs: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"


@dataclass
class DataConfig:
    alpaca_raw_path: str
    alpaca_train_path: str
    alpaca_eval_path: str
    teacher_prompt_seed_path: str
    teacher_train_path: str
    teacher_eval_path: str
    sample_size: int = 10
    random_seed: int = 42


@dataclass
class EvalConfig:
    output_root: str
    checkpoints: list[str]
    judge_dimensions: list[str]
    swap_order: bool = True
    max_prompts: int = 100


@dataclass
class RuntimeConfig:
    artifacts_root: str
    prompts_dir: str
    project_root: str = "."
    use_mock_backends: bool = True
    full_run_guard_file: str = "artifacts/full_run_completed.flag"


@dataclass
class ProjectConfig:
    experiment_name: str
    models: ModelConfig
    data: DataConfig
    stage1: TrainingStageConfig
    stage2: TrainingStageConfig
    evaluation: EvalConfig
    runtime: RuntimeConfig
    extras: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = _expand_env_values(yaml.safe_load(handle))
    return ProjectConfig(
        experiment_name=raw["experiment_name"],
        models=ModelConfig(**raw["models"]),
        data=DataConfig(**raw["data"]),
        stage1=TrainingStageConfig(name="stage1", **raw["stage1"]),
        stage2=TrainingStageConfig(name="stage2", **raw["stage2"]),
        evaluation=EvalConfig(**raw["evaluation"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        extras=raw.get("extras", {}),
    )
