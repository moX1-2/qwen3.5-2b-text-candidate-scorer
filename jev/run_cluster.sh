#!/usr/bin/env bash
# 一键后台启动 A100 专享算力训练脚本
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 默认输出目录
output_dir="runs/cluster-a100"

args=()
has_output_dir=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            output_dir="$2"
            args+=("$1" "$2")
            has_output_dir=1
            shift 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

if [ $has_output_dir -eq 0 ]; then
    args=("--output-dir" "$output_dir" "${args[@]}")
fi

mkdir -p "$output_dir"

echo "============================================================"
echo "🚀 启动 A100 专享算力微调任务 (后台静默运行模式)"
echo "============================================================"

# 检查当前是否已有任务在运行
if [ -f "$output_dir/train.pid" ]; then
    old_pid=$(cat "$output_dir/train.pid" 2>/dev/null || true)
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        echo "⚠️ 警告: 已有正在运行的训练进程 (PID: $old_pid)！"
        echo "如需停止旧进程，请运行: kill $old_pid"
        exit 1
    fi
fi

# 确保激活了 conda 环境（如果存在）
if [ -d "$HOME/miniconda3/bin" ]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
    eval "$(conda shell.bash hook 2>/dev/null || true)"
    conda activate jev 2>/dev/null || true
fi

# 后台静默启动，默认开启自动断点续传 (--resume auto)
nohup python3 -u train_cluster.py   --resume auto   --epochs 2   --gradient-accumulation 16   --micro-batch-size 4   --eval-every 100   --save-every 50   --log-every 10   "${args[@]}" > "$output_dir/stdout.log" 2>&1 &

pid=$!
echo "$pid" > "$output_dir/train.pid"

echo "✅ 训练已成功转入后台运行 (PID: $pid)"
echo "📁 输出目录: $output_dir/"
echo "🔄 自动续传: 已开启 (--resume auto，自动读取该目录下 latest.json)"
echo "📜 原始日志: tail -f $output_dir/train.log"
echo "📊 启动专属动态监控: python3 monitor.py --output-dir $output_dir --watch"
echo "📸 查看单次快照状态: python3 monitor.py --output-dir $output_dir"
echo "⏹️ 终止后台任务: kill $pid (或 kill -SIGINT $pid 安全落盘后退出)"
echo "============================================================"
