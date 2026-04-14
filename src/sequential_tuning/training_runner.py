from __future__ import annotations

import argparse
import os
from pathlib import Path

from sequential_tuning.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal QLoRA runner placeholder for UTSA HPC execution.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--hf_token_env", default="HF_TOKEN")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary = vars(args)
    summary["status"] = "dry_run_only" if args.dry_run else "starting_training"
    write_json(summary, Path(args.output_dir) / "training_runner_args.json")
    if args.dry_run:
        print(f"Prepared dry-run training job in {args.output_dir}")
        return

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    hf_token = os.getenv(args.hf_token_env)

    def infer_target_modules(model_name: str) -> list[str]:
        lowered = model_name.lower()
        if "phi-3" in lowered or "phi3" in lowered:
            return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        if "llama" in lowered or "qwen" in lowered or "mistral" in lowered:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def format_example(example: dict) -> str:
        input_text = example.get("input", "").strip()
        return (
            f"### Instruction:\n{example['instruction'].strip()}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{example['output'].strip()}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and torch.cuda.is_bf16_supported()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    peft_config = None
    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(
            model,
            args.resume_from_checkpoint,
            is_trainable=True,
        )
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=infer_target_modules(args.model_name),
        )

    train_dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=bf16_supported,
        fp16=cuda_available and not bf16_supported,
        report_to="none",
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=format_example,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    summary["status"] = "completed"
    write_json(summary, Path(args.output_dir) / "training_runner_args.json")
    print(f"Training complete: {args.output_dir}")


if __name__ == "__main__":
    main()

