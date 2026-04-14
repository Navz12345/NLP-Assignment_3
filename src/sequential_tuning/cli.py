from __future__ import annotations

import argparse
from pathlib import Path

from sequential_tuning.config import load_config
from sequential_tuning.pipelines.aggregate import build_results_table
from sequential_tuning.pipelines.evaluate import evaluate_alpaca_predictions, evaluate_json_predictions, run_pairwise_judge
from sequential_tuning.pipelines.human_seed_writer import build_human_seed_dataset
from sequential_tuning.pipelines.inference import run_inference
from sequential_tuning.pipelines.prepare_data import prepare_alpaca_data, prepare_teacher_seed_prompts
from sequential_tuning.pipelines.seed_builder import build_full_json_prompt_seed_dataset
from sequential_tuning.pipelines.teacher_data import generate_teacher_dataset
from sequential_tuning.pipelines.train import create_placeholder_adapter, materialize_stage_artifacts
from sequential_tuning.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential instruction tuning project CLI")
    parser.add_argument("--config", default="configs/sample.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-data")
    seed_parser = subparsers.add_parser("build-json-seeds")
    seed_parser.add_argument("--output", default="data/raw/json_prompt_seeds_full.json")
    seed_parser.add_argument("--prompts-per-task", type=int, default=24)
    seed_parser.add_argument("--eval-per-task", type=int, default=4)
    human_seed_parser = subparsers.add_parser("build-human-json-seeds")
    human_seed_parser.add_argument("--output", default="data/raw/json_prompt_seeds_human_authored.json")
    human_seed_parser.add_argument("--train-per-task", type=int, default=40)
    human_seed_parser.add_argument("--eval-per-task", type=int, default=20)
    subparsers.add_parser("generate-teacher-data")

    train_parser = subparsers.add_parser("prepare-training")
    train_parser.add_argument("--stage", choices=["stage1", "stage2", "all"], default="all")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint", required=True)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", required=True)

    judge_parser = subparsers.add_parser("judge")
    judge_parser.add_argument("--task-group", choices=["alpaca", "json"], required=True)
    judge_parser.add_argument("--checkpoint-a", required=True)
    judge_parser.add_argument("--checkpoint-b", required=True)

    subparsers.add_parser("aggregate")
    sanity_parser = subparsers.add_parser("sanity-check")
    sanity_parser.add_argument("--full", action="store_true")
    full_run_parser = subparsers.add_parser("full-run")
    full_run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("quickstart")
    return parser


def command_prepare_data(config_path: str) -> None:
    config = load_config(config_path)
    print(prepare_alpaca_data(config))
    print(prepare_teacher_seed_prompts(config))


def command_build_json_seeds(output: str, prompts_per_task: int, eval_per_task: int) -> None:
    print(
        build_full_json_prompt_seed_dataset(
            output_path=output,
            prompts_per_task=prompts_per_task,
            eval_per_task=eval_per_task,
        )
    )


def command_build_human_json_seeds(output: str, train_per_task: int, eval_per_task: int) -> None:
    print(build_human_seed_dataset(output, train_per_task=train_per_task, eval_per_task=eval_per_task))


def command_generate_teacher_data(config_path: str) -> None:
    config = load_config(config_path)
    print(generate_teacher_dataset(config))


def command_prepare_training(config_path: str, stage: str) -> None:
    config = load_config(config_path)
    if stage in {"stage1", "all"}:
        print(materialize_stage_artifacts(config, config.stage1))
        create_placeholder_adapter(config.stage1)
    if stage in {"stage2", "all"}:
        print(materialize_stage_artifacts(config, config.stage2, base_checkpoint=config.stage1.output_dir))
        create_placeholder_adapter(config.stage2)


def command_infer(config_path: str, checkpoint: str) -> None:
    config = load_config(config_path)
    run_inference(config, config.data.alpaca_eval_path, checkpoint, "alpaca")
    run_inference(config, config.data.teacher_eval_path, checkpoint, "json")
    print(f"Inference artifacts written for {checkpoint}")


def command_evaluate(config_path: str, checkpoint: str) -> None:
    config = load_config(config_path)
    checkpoint_dir = Path(config.evaluation.output_root) / checkpoint
    print(evaluate_alpaca_predictions(config, checkpoint_dir / "alpaca_predictions.jsonl", checkpoint))
    print(evaluate_json_predictions(config, checkpoint_dir / "json_predictions.jsonl", checkpoint))


def command_judge(config_path: str, task_group: str, checkpoint_a: str, checkpoint_b: str) -> None:
    config = load_config(config_path)
    base = Path(config.evaluation.output_root)
    print(
        run_pairwise_judge(
            config,
            base / checkpoint_a / f"{task_group}_predictions.jsonl",
            base / checkpoint_b / f"{task_group}_predictions.jsonl",
            task_group,
            checkpoint_a,
            checkpoint_b,
        )
    )


def command_aggregate(config_path: str) -> None:
    config = load_config(config_path)
    print(build_results_table(config))


def command_quickstart(config_path: str) -> None:
    command_prepare_data(config_path)
    command_generate_teacher_data(config_path)
    command_prepare_training(config_path, "all")
    for checkpoint in ["checkpoint_0", "checkpoint_1", "checkpoint_2"]:
        command_infer(config_path, checkpoint)
        command_evaluate(config_path, checkpoint)
    command_judge(config_path, "alpaca", "checkpoint_0", "checkpoint_1")
    command_judge(config_path, "alpaca", "checkpoint_1", "checkpoint_2")
    command_judge(config_path, "json", "checkpoint_0", "checkpoint_2")
    command_aggregate(config_path)


def command_sanity_check(config_path: str, full: bool = False) -> None:
    config = load_config(config_path)
    checks = {
        "config_loaded": True,
        "alpaca_source_exists": Path(config.data.alpaca_raw_path).exists(),
        "teacher_seed_exists": Path(config.data.teacher_prompt_seed_path).exists(),
        "teacher_prompt_exists": Path(config.runtime.prompts_dir, "teacher_system_prompt.txt").exists(),
        "judge_prompt_exists": Path(config.runtime.prompts_dir, "judge_prompt.txt").exists(),
        "full_mode": full,
    }
    if full:
        checks["mock_backends_disabled"] = not config.runtime.use_mock_backends
        checks["hpc_stage1_script_exists"] = Path("hpc/train_stage1.slurm").exists()
        checks["hpc_stage2_script_exists"] = Path("hpc/train_stage2.slurm").exists()
    output_path = Path(config.runtime.artifacts_root) / "sanity_check.json"
    write_json(checks, output_path)
    print(checks)


def command_full_run(config_path: str, force: bool = False) -> None:
    config = load_config(config_path)
    guard_path = Path(config.runtime.full_run_guard_file)
    if guard_path.exists() and not force:
        raise SystemExit(f"Full run already marked complete at {guard_path}. Use --force to rerun.")
    command_prepare_data(config_path)
    command_generate_teacher_data(config_path)
    command_prepare_training(config_path, "all")
    for checkpoint in config.evaluation.checkpoints:
        command_infer(config_path, checkpoint)
        command_evaluate(config_path, checkpoint)
    command_judge(config_path, "alpaca", "checkpoint_0", "checkpoint_1")
    command_judge(config_path, "alpaca", "checkpoint_1", "checkpoint_2")
    command_judge(config_path, "json", "checkpoint_0", "checkpoint_2")
    command_aggregate(config_path)
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    guard_path.write_text("completed\n", encoding="utf-8")
    print(f"Full run completed and guard file written to {guard_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare-data":
        command_prepare_data(args.config)
    elif args.command == "build-json-seeds":
        command_build_json_seeds(args.output, args.prompts_per_task, args.eval_per_task)
    elif args.command == "build-human-json-seeds":
        command_build_human_json_seeds(args.output, args.train_per_task, args.eval_per_task)
    elif args.command == "generate-teacher-data":
        command_generate_teacher_data(args.config)
    elif args.command == "prepare-training":
        command_prepare_training(args.config, args.stage)
    elif args.command == "infer":
        command_infer(args.config, args.checkpoint)
    elif args.command == "evaluate":
        command_evaluate(args.config, args.checkpoint)
    elif args.command == "judge":
        command_judge(args.config, args.task_group, args.checkpoint_a, args.checkpoint_b)
    elif args.command == "aggregate":
        command_aggregate(args.config)
    elif args.command == "sanity-check":
        command_sanity_check(args.config, full=args.full)
    elif args.command == "full-run":
        command_full_run(args.config, force=args.force)
    elif args.command == "quickstart":
        command_quickstart(args.config)


if __name__ == "__main__":
    main()

