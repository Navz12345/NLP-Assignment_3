#!/bin/bash
#SBATCH --job-name=stage1_alpaca_qlora
#SBATCH --output=artifacts/logs/stage1_%j.out
#SBATCH --error=artifacts/logs/stage1_%j.err
#SBATCH --partition=gpu1a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:039
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00

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
  --dataset_path data/processed/alpaca_train.jsonl \
  --output_dir artifacts/checkpoints/stage1 \
  --epochs 2 \
  --learning_rate 2e-5 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_seq_length 2048 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --hf_token_env HF_TOKEN

