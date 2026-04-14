from __future__ import annotations

import json
from collections import Counter
from typing import Any


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def parse_json_safe(text: str) -> tuple[bool, Any | None, str | None]:
    try:
        cleaned = _clean_json_text(text)
        return True, json.loads(cleaned), None
    except json.JSONDecodeError as exc:
        return False, None, str(exc)


def infer_error_label(error_message: str | None) -> str:
    if not error_message:
        return "unknown"
    lowered = error_message.lower()
    if "expecting ',' delimiter" in lowered:
        return "missing_comma_or_bracket"
    if "unterminated string" in lowered:
        return "unterminated_string"
    if "extra data" in lowered:
        return "extra_trailing_content"
    if "expecting value" in lowered:
        return "missing_value"
    return "other_parse_error"


def schema_compliant(candidate: Any, schema: dict[str, str]) -> bool:
    if not isinstance(candidate, dict):
        return False
    for key, expected_type in schema.items():
        if key not in candidate:
            return False
        if expected_type == "string" and not isinstance(candidate[key], str):
            return False
        if expected_type == "number" and not isinstance(candidate[key], (int, float)):
            return False
        if expected_type == "boolean" and not isinstance(candidate[key], bool):
            return False
        if expected_type == "array" and not isinstance(candidate[key], list):
            return False
        if expected_type == "object" and not isinstance(candidate[key], dict):
            return False
    return True


def flat_field_f1(reference: dict[str, Any], prediction: dict[str, Any]) -> dict[str, float]:
    ref_pairs = {(key, json.dumps(value, sort_keys=True)) for key, value in reference.items()}
    pred_pairs = {(key, json.dumps(value, sort_keys=True)) for key, value in prediction.items()}
    tp = len(ref_pairs & pred_pairs)
    fp = len(pred_pairs - ref_pairs)
    fn = len(ref_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize_error_taxonomy(labels: list[str]) -> dict[str, int]:
    return dict(Counter(labels))

