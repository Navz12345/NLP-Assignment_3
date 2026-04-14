from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    prompt_id: str
    split: str
    task_type: str
    instruction: str
    input: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    prompt_id: str
    checkpoint: str
    output_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    prompt_id: str
    checkpoint_a: str
    checkpoint_b: str
    response_a_scores: dict[str, float]
    response_b_scores: dict[str, float]
    winner: str
    justification: str

