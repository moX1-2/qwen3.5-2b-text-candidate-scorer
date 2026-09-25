#!/usr/bin/env bash
#SBATCH --job-name=qwen-jev-a100
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --output=runs/cluster-a100/slurm-%j.out
#SBATCH --error=runs/cluster-a100/slurm-%j.err

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

mkdir -p runs/cluster-a100

# 激活环境 (优先使用 miniconda)
if [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/bin/activate" jev
elif command -v conda >/dev/null 2>&1; then
    conda activate jev || true
fi

echo "=========================================================="
echo "Slurm Job ID: $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPU Allocated: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo Unknown)"
echo "=========================================================="

python3 -u train_cluster.py   --output-dir runs/cluster-a100   --resume auto   --epochs 2   --gradient-accumulation 16   --micro-batch-size 4   --eval-every 100   --save-every 50   --log-every 10
