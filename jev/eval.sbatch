#!/usr/bin/env bash
#SBATCH --job-name=jev-zero-shot
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:50:00
#SBATCH --output=jev-eval-%j.out
#SBATCH --error=jev-eval-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

python_bin=$(printenv JEV_PYTHON || true)
if [ -z "$python_bin" ]; then
    python_bin="$PWD/.venv/bin/python"
fi
if [ ! -x "$python_bin" ]; then
    echo "Python environment not found: $python_bin" >&2
    exit 1
fi

nvidia-smi
"$python_bin" evaluate_zero_shot.py \
    --model ../models/Qwen3.5-2B-Text \
    --data ../datasets/selected/validation.jsonl \
    --output-dir runs/zero-shot-full \
    --styles plain chat \
    --orders original reverse
