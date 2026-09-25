"""面向算力集群后台任务的专属监控工具 (Cluster Monitor).

特性:
- 随时启动，随时退出 (按 Ctrl + C 或 q 退出，绝对不影响后台训练运行)。
- 支持单次快照模式: python monitor.py (打一行或一张表格即退)。
- 支持动态观察模式: python monitor.py --watch (每 2 秒轻量刷新一次，展示步数、Loss、吞吐速度、真机显存与利用率)。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


def get_gpu_info() -> dict[str, str]:
    if not shutil.which("nvidia-smi"):
        return {"name": "未知 (无 nvidia-smi)", "util": "N/A", "mem": "N/A", "power": "N/A", "temp": "N/A"}
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, encoding="utf-8", timeout=3).strip()
        lines = out.splitlines()
        if not lines:
            return {"name": "N/A", "util": "N/A", "mem": "N/A", "power": "N/A", "temp": "N/A"}
        parts = [p.strip() for p in lines[0].split(",")]
        name = parts[0]
        util = f"{parts[1]}%"
        mem = f"{float(parts[2])/1024:.1f}G / {float(parts[3])/1024:.1f}G"
        power = f"{float(parts[4]):.0f}W / {float(parts[5]):.0f}W"
        temp = f"{parts[6]}°C"
        return {"name": name, "util": util, "mem": mem, "power": power, "temp": temp}
    except Exception:
        return {"name": "N/A", "util": "N/A", "mem": "N/A", "power": "N/A", "temp": "N/A"}


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_latest_metrics(output_dir: Path) -> dict[str, Any] | None:
    metric_file = output_dir / "metrics.jsonl"
    if not metric_file.exists():
        return None
    try:
        with metric_file.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            buffer_size = min(size, 8192)
            f.seek(size - buffer_size)
            lines = f.read().decode("utf-8", errors="ignore").splitlines()
            for line in reversed(lines):
                line = line.strip()
                if line and "step" in line:
                    return json.loads(line)
    except Exception:
        pass
    return None


def read_recent_logs(output_dir: Path, count: int = 4) -> list[str]:
    log_file = output_dir / "train.log"
    if not log_file.exists():
        return ["等待日志输出..."]
    try:
        with log_file.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            buffer_size = min(size, 8192)
            f.seek(size - buffer_size)
            lines = f.read().decode("utf-8", errors="ignore").splitlines()
            valid = [l.strip() for l in lines if l.strip()]
            return valid[-count:] if valid else ["日志为空"]
    except Exception:
        return ["读取日志异常"]


def render_dashboard(output_dir: Path, gpu_info: dict[str, str], is_running: bool, pid: int | None) -> Panel:
    latest = read_latest_metrics(output_dir)
    recent_logs = read_recent_logs(output_dir, count=4)

    status_str = f"[bold green]● 运行中 (PID: {pid})[/bold green]" if is_running else "[bold yellow]○ 未在运行 (已结束或待启动)[/bold yellow]"
    header_content = (
        f"任务目录: {output_dir.resolve()} | 状态: {status_str}\n"
        f"硬件设备: {gpu_info['name']} | 算力利用率: {gpu_info['util']} | 温度: {gpu_info['temp']} | 功耗: {gpu_info['power']}"
    )

    metrics_table = Table(box=box.ROUNDED, expand=True)
    metrics_table.add_column("当前步数 / 进度", justify="center")
    metrics_table.add_column("训练 Loss", justify="center")
    metrics_table.add_column("Batch 准确率", justify="center")
    metrics_table.add_column("速度 / 吞吐", justify="center")
    metrics_table.add_column("显存占用", justify="center")

    if latest:
        step_val = f"{latest.get('step', 0)} / {latest.get('total_steps', 0)} ({latest.get('progress_percent', 0.0):.1f}%)"
        loss_val = f"{latest.get('loss', 0.0):.4f}"
        acc_val = f"{latest.get('accuracy', 0.0)*100:.1f}%"
        speed_val = f"{latest.get('step_seconds', 0.0):.2f}s/step ({latest.get('samples_per_sec', 0.0):.1f}题/s)"
        vram_val = f"{latest.get('vram_gib', 0.0):.1f}G ({gpu_info['mem']})"
    else:
        step_val, loss_val, acc_val, speed_val, vram_val = "等待中...", "--", "--", "--", gpu_info['mem']

    metrics_table.add_row(step_val, loss_val, acc_val, speed_val, vram_val)

    log_box = Panel("\n".join(recent_logs), title="最新日志 (tail -n 4)", box=box.ROUNDED, style="dim")

    return Panel(
        Group(Panel(header_content, box=box.ROUNDED, style="bold"), metrics_table, log_box),
        title="[bold]JEV 算力集群后台任务监控看板[/bold]",
        subtitle="按 Ctrl + C 随时退出监控 (绝对不会影响后台训练)",
        box=box.DOUBLE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="JEV 算力集群后台监控器")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cluster-a100"), help="训练输出目录")
    parser.add_argument("--watch", "--follow", "-w", action="store_true", help="动态刷新监控 (每 2 秒刷新一次)")
    args = parser.parse_args()

    console = Console()
    pid_file = args.output_dir / "train.pid"

    if not args.watch:
        # 单次快照输出
        pid = int(pid_file.read_text(encoding="utf-8").strip()) if pid_file.exists() else None
        alive = is_pid_alive(pid) if pid else False
        gpu = get_gpu_info()
        console.print(render_dashboard(args.output_dir, gpu, alive, pid))
        return

    # 动态看板循环刷新
    try:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                pid = int(pid_file.read_text(encoding="utf-8").strip()) if pid_file.exists() else None
                alive = is_pid_alive(pid) if pid else False
                gpu = get_gpu_info()
                live.update(render_dashboard(args.output_dir, gpu, alive, pid))
                time.sleep(2)
    except KeyboardInterrupt:
        console.print("\n[dim]已退出监控模式，后台训练仍在独立运行。[/dim]")


if __name__ == "__main__":
    main()
