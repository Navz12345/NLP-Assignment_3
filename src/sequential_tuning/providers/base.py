from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


class TextGenerationProvider:
    def generate(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 512) -> str:
        raise NotImplementedError

