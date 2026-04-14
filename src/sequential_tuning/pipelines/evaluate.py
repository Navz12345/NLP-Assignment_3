from __future__ import annotations

import json
from pathlib import Path

from sequential_tuning.config import ProjectConfig
from sequential_tuning.providers.base import Message
from sequential_tuning.providers.factory import build_judge_provider
from sequential_tuning.utils.io import read_jsonl, write_json
from sequential_tuning.utils.json_eval import flat_field_f1, infer_error_label, parse_json_safe, schema_compliant, summarize_error_taxonomy
from sequential_tuning.utils.metrics import average, overlap_f1, token_count, try_compute_bertscore_f1, try_compute_rouge_l
from sequential_tuning.utils.templates import load_template


def evaluate_alpaca_predictions(config: ProjectConfig, predictions_path: str, checkpoint_name: str) -> dict:
    rows = read_jsonl(predictions_path)
    scores = [overlap_f1(row["reference_output"], row["prediction"]) for row in rows]
    references = [row["reference_output"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    results = {
        "checkpoint": checkpoint_name,
        "avg_overlap_f1": average(scores),
        "rouge_l": try_compute_rouge_l(references, predictions),
        "bertscore_f1": try_compute_bertscore_f1(references, predictions),
        "avg_output_tokens": average(token_count(row["prediction"]) for row in rows),
        "task_completion_rate": average(1.0 if row["prediction"].strip() else 0.0 for row in rows),
        "count": len(rows),
    }
    output_path = Path(config.evaluation.output_root) / checkpoint_name / "alpaca_metrics.json"
    write_json(results, output_path)
    return results


def evaluate_json_predictions(config: ProjectConfig, predictions_path: str, checkpoint_name: str) -> dict:
    rows = read_jsonl(predictions_path)
    validity = []
    compliance = []
    exact_match = []
    field_scores = []
    errors = []
    for row in rows:
        valid, parsed, error = parse_json_safe(row["prediction"])
        validity.append(1.0 if valid else 0.0)
        if not valid:
            errors.append(infer_error_label(error))
            compliance.append(0.0)
            exact_match.append(0.0)
            field_scores.append(0.0)
            continue
        reference_valid, reference_parsed, _ = parse_json_safe(row["reference_output"])
        schema = row.get("metadata", {}).get("schema", {})
        compliance.append(1.0 if schema_compliant(parsed, schema) else 0.0)
        exact_match.append(1.0 if reference_valid and parsed == reference_parsed else 0.0)
        if reference_valid and isinstance(reference_parsed, dict) and isinstance(parsed, dict):
            field_scores.append(flat_field_f1(reference_parsed, parsed)["f1"])
        else:
            field_scores.append(0.0)
    results = {
        "checkpoint": checkpoint_name,
        "json_validity_rate": average(validity),
        "schema_compliance_rate": average(compliance),
        "exact_match_rate": average(exact_match),
        "field_level_f1": average(field_scores),
        "error_taxonomy": summarize_error_taxonomy(errors),
        "count": len(rows),
    }
    output_path = Path(config.evaluation.output_root) / checkpoint_name / "json_metrics.json"
    write_json(results, output_path)
    return results


def run_pairwise_judge(config: ProjectConfig, prediction_path_a: str, prediction_path_b: str, task_group: str, checkpoint_a: str, checkpoint_b: str) -> dict:
    rows_a = read_jsonl(prediction_path_a)
    rows_b = {row["prompt_id"]: row for row in read_jsonl(prediction_path_b)}
    provider = build_judge_provider(config)
    prompt_template = load_template(config.runtime.prompts_dir, "judge_prompt.txt")
    dimensions = config.evaluation.judge_dimensions
    results = []
    wins = {"A": 0, "B": 0, "tie": 0}
    for row_a in rows_a:
        row_b = rows_b[row_a["prompt_id"]]
        prompt = prompt_template.format(
            prompt_id=row_a["prompt_id"],
            task_group=task_group,
            dimensions=", ".join(dimensions),
            instruction=row_a["instruction"],
            input_text=row_a["input"],
            response_a=row_a["prediction"],
            response_b=row_b["prediction"],
        )
        response = provider.generate([Message(role="user", content=prompt)], max_tokens=512).strip()

        if response.startswith("```"):
            response = response.strip("`")
            if response.lower().startswith("json"):
                response = response[4:].strip()

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = {
                "response_a_scores": {dim: 0 for dim in dimensions},
                "response_b_scores": {dim: 0 for dim in dimensions},
                "winner": "tie",
                "justification": f"Judge returned non-JSON output: {response[:200]}",
            }

        winner = parsed.get("winner", "tie")
        wins[winner] = wins.get(winner, 0) + 1
        results.append(
            {
                "prompt_id": row_a["prompt_id"],
                "checkpoint_a": checkpoint_a,
                "checkpoint_b": checkpoint_b,
                **parsed,
            }
        )
    summary = {
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "task_group": task_group,
        "wins": wins,
        "count": len(results),
        "win_rate_a": wins["A"] / len(results) if results else 0.0,
        "win_rate_b": wins["B"] / len(results) if results else 0.0,
        "tie_rate": wins["tie"] / len(results) if results else 0.0,
    }
    output_dir = Path(config.evaluation.output_root) / "judge"
    write_json(results, output_dir / f"{task_group}_{checkpoint_a}_vs_{checkpoint_b}_details.json")
    write_json(summary, output_dir / f"{task_group}_{checkpoint_a}_vs_{checkpoint_b}_summary.json")
    return summary
