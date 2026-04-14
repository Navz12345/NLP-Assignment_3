from __future__ import annotations

from pathlib import Path

from sequential_tuning.config import ProjectConfig
from sequential_tuning.utils.io import read_json, write_json


def build_results_table(config: ProjectConfig) -> dict:
    checkpoints = config.evaluation.checkpoints
    rows = []
    for checkpoint in checkpoints:
        checkpoint_dir = Path(config.evaluation.output_root) / checkpoint
        alpaca_metrics = read_json(checkpoint_dir / "alpaca_metrics.json")
        json_metrics = read_json(checkpoint_dir / "json_metrics.json")
        rows.append(
            {
                "checkpoint": checkpoint,
                "alpaca_avg_overlap_f1": alpaca_metrics["avg_overlap_f1"],
                "alpaca_rouge_l": alpaca_metrics["rouge_l"],
                "alpaca_bertscore_f1": alpaca_metrics["bertscore_f1"],
                "alpaca_task_completion_rate": alpaca_metrics["task_completion_rate"],
                "json_validity_rate": json_metrics["json_validity_rate"],
                "schema_compliance_rate": json_metrics["schema_compliance_rate"],
                "exact_match_rate": json_metrics["exact_match_rate"],
                "field_level_f1": json_metrics["field_level_f1"],
            }
        )
    forgetting = {}
    by_checkpoint = {row["checkpoint"]: row for row in rows}
    if "checkpoint_1" in by_checkpoint and "checkpoint_2" in by_checkpoint:
        forgetting = {
            "alpaca_overlap_f1_change": by_checkpoint["checkpoint_2"]["alpaca_avg_overlap_f1"] - by_checkpoint["checkpoint_1"]["alpaca_avg_overlap_f1"],
            "alpaca_rouge_l_change": (by_checkpoint["checkpoint_2"]["alpaca_rouge_l"] or 0.0) - (by_checkpoint["checkpoint_1"]["alpaca_rouge_l"] or 0.0),
            "alpaca_bertscore_f1_change": (by_checkpoint["checkpoint_2"]["alpaca_bertscore_f1"] or 0.0) - (by_checkpoint["checkpoint_1"]["alpaca_bertscore_f1"] or 0.0),
            "json_validity_change": by_checkpoint["checkpoint_2"]["json_validity_rate"] - by_checkpoint["checkpoint_1"]["json_validity_rate"],
        }
    report = {"rows": rows, "forgetting_analysis": forgetting}
    write_json(report, Path(config.evaluation.output_root) / "summary_table.json")
    return report
