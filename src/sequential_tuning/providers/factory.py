from __future__ import annotations

import os

from sequential_tuning.config import ProjectConfig
from sequential_tuning.providers.base import TextGenerationProvider
from sequential_tuning.providers.local_hf import LocalHFProvider
from sequential_tuning.providers.mock_provider import MockProvider
from sequential_tuning.providers.openai_compatible import OpenAICompatibleProvider


def build_teacher_provider(config: ProjectConfig) -> TextGenerationProvider:
    if config.models.teacher_provider == "mock" or config.runtime.use_mock_backends:
        return MockProvider(mode="teacher")
    if config.models.teacher_provider == "openai_compatible":
        return OpenAICompatibleProvider(
            model_name=config.models.teacher_model_name,
            base_url_env=config.models.teacher_base_url_env,
            api_key_env=config.models.teacher_api_key_env,
        )
    raise ValueError(f"Unsupported teacher provider: {config.models.teacher_provider}")


def build_judge_provider(config: ProjectConfig) -> TextGenerationProvider:
    if config.models.judge_provider == "mock" or config.runtime.use_mock_backends:
        return MockProvider(mode="judge")
    if config.models.judge_provider == "openai_compatible":
        return OpenAICompatibleProvider(
            model_name=config.models.judge_model_name,
            base_url_env=config.models.judge_base_url_env,
            api_key_env=config.models.judge_api_key_env,
        )
    raise ValueError(f"Unsupported judge provider: {config.models.judge_provider}")


def build_inference_provider(config: ProjectConfig, adapter_path: str | None = None) -> TextGenerationProvider:
    if config.models.inference_provider == "mock" or config.runtime.use_mock_backends:
        return MockProvider(mode="student")
    if config.models.inference_provider == "openai_compatible":
        return OpenAICompatibleProvider(
            model_name=config.models.inference_model_name,
            base_url_env=config.models.inference_base_url_env,
            api_key_env=config.models.inference_api_key_env,
        )
    if config.models.inference_provider == "hf_local":
        return LocalHFProvider(
            base_model_name=config.models.inference_model_name,
            adapter_path=adapter_path,
            hf_token=os.getenv(config.models.student_hf_token_env),
        )
    raise ValueError(f"Unsupported inference provider: {config.models.inference_provider}")
