#!/bin/bash
#SBATCH --job-name=stage2_json_qlora
#SBATCH --output=artifacts/logs/stage2_%j.out
#SBATCH --error=artifacts/logs/stage2_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --nodelist=gpu011
#SBATCH --time=02:00:00

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/$USER/NLP_Assignment3_Navya/Playground/env
cd "$SLURM_SUBMIT_DIR"
mkdir -p artifacts/logs

export HF_HOME=/work/$USER/.HF_cache
export TRANSFORMERS_CACHE=/work/$USER/.HF_cache

: "${HF_TOKEN:?HF_TOKEN must be exported before submitting this job}"

python -m sequential_tuning.training_runner \
  --model_name microsoft/Phi-3.5-mini-instruct \
  --dataset_path data/processed/teacher_train.jsonl \
  --output_dir artifacts/checkpoints/stage2 \
  --epochs 2 \
  --learning_rate 1e-5 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_seq_length 2048 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --hf_token_env HF_TOKEN \
  --resume_from_checkpoint artifacts/checkpoints/stage1

