from __future__ import annotations

import os

from sequential_tuning.providers.base import Message, TextGenerationProvider


class OpenAICompatibleProvider(TextGenerationProvider):
    def __init__(self, model_name: str, base_url_env: str = "OPENAI_BASE_URL", api_key_env: str = "OPENAI_API_KEY") -> None:
        from openai import OpenAI

        base_url = os.getenv(base_url_env)
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable: {api_key_env}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": message.role, "content": message.content} for message in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
