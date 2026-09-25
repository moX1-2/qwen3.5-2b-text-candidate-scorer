#!/usr/bin/env bash
#SBATCH --job-name=jev-2b-lora
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:50:00
#SBATCH --output=jev-train-%j.out
#SBATCH --error=jev-train-%j.err
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
method=$(printenv JEV_METHOD || true)
if [ -z "$method" ]; then
    method=lora
fi
output_dir=$(printenv JEV_OUTPUT_DIR || true)
if [ -z "$output_dir" ]; then
    output_dir="$PWD/runs/lora-pilot"
fi

set -- \
    --model ../models/Qwen3.5-2B-Text \
    --train-data ../datasets/selected/train.jsonl \
    --validation-data ../datasets/selected/validation.jsonl \
    --output-dir "$output_dir" \
    --method "$method" \
    --epochs 2 \
    --max-length 1024 \
    --gradient-accumulation 16
max_steps=$(printenv JEV_MAX_STEPS || true)
if [ -n "$max_steps" ]; then
    set -- "$@" --max-steps "$max_steps"
fi
if [ -f "$output_dir/latest.json" ]; then
    set -- "$@" --resume auto
fi

nvidia-smi
"$python_bin" train.py "$@"
