from __future__ import annotations

import json

from sequential_tuning.providers.base import Message, TextGenerationProvider


class MockProvider(TextGenerationProvider):
    def __init__(self, mode: str = "teacher") -> None:
        self.mode = mode

    def generate(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 512) -> str:
        content = messages[-1].content.lower()
        if self.mode == "judge":
            return json.dumps(
                {
                    "response_a_scores": {
                        "instruction_following": 4,
                        "correctness": 4,
                        "clarity": 4,
                        "completeness": 4,
                        "structured_output_validity": 4,
                        "hallucination_risk": 4,
                    },
                    "response_b_scores": {
                        "instruction_following": 3,
                        "correctness": 3,
                        "clarity": 3,
                        "completeness": 3,
                        "structured_output_validity": 3,
                        "hallucination_risk": 3,
                    },
                    "winner": "A",
                    "justification": "Mock judge preferred response A.",
                }
            )
        if "repair" in content:
            return json.dumps({"fixed_json": {"status": "repaired", "valid": True}})
        if "classification" in content:
            return json.dumps({"label": "support", "confidence": 0.92})
        if "tool" in content:
            return json.dumps({"tool_name": "search_hotels", "arguments": {"city": "Austin", "nights": 2}})
        if "schema" in content:
            return json.dumps({"title": "Example", "priority": "high", "done": False})
        return json.dumps({"entities": [{"type": "person", "text": "Ada Lovelace"}], "dates": ["1843-01-01"]})

