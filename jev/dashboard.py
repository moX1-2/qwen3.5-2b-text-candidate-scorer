"""纯黑白素雅风格终端训练监控看板 (Monochrome Terminal Dashboard)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table


class TrainDashboard:
    """纯黑白素雅终端训练监控看板."""

    def __init__(
        self,
        total_examples: int,
        total_steps: int,
        method: str,
        model_name: str,
        gradient_accumulation: int,
    ) -> None:
        self.console = Console(highlight=False, no_color=True)
        self.total_examples = total_examples
        self.total_steps = total_steps
        self.method = method.upper()
        self.model_name = model_name
        self.gradient_accumulation = gradient_accumulation

        # 核心指标
        self.current_step = 0
        self.examples_seen = 0
        self.current_loss = 0.0
        self.initial_loss: float | None = None
        self.current_acc = 0.0
        self.current_lr = 0.0
        self.peak_gpu = 0.0
        self.val_acc: float | None = None
        self.val_nll: float | None = None

        # 纯黑白进度条 (无彩色，依靠实心/暗淡线条对比)
        self.progress = Progress(
            TextColumn("{task.description}", style="bold"),
            BarColumn(bar_width=28, style="dim", complete_style="bold", finished_style="bold"),
            TextColumn("{task.percentage:>3.0f}%", style="bold"),
            TextColumn("[{task.completed}/{task.total}]", style="dim"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.task_examples: TaskID = self.progress.add_task(
            "题目进度", total=total_examples
        )
        self.task_steps: TaskID = self.progress.add_task(
            "Step 迭代", total=total_steps
        )

        self.logs: list[str] = [
            f"[{datetime.now().strftime('%H:%M:%S')}] 训练已就绪 | 目标: {total_examples} 题 ({total_steps} Steps)"
        ]
        self.live: Live | None = None

    def start(self) -> None:
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self.live.start()

    def stop(self) -> None:
        if self.live:
            self.live.update(self._render())
            self.live.stop()

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {message}")
        if len(self.logs) > 4:
            self.logs.pop(0)

    def update_step(
        self,
        step: int,
        examples_seen: int,
        loss: float,
        acc: float,
        lr: float,
        peak_gpu: float,
    ) -> None:
        self.current_step = step
        self.examples_seen = examples_seen
        self.current_loss = loss
        if self.initial_loss is None and loss > 0:
            self.initial_loss = loss
        self.current_acc = acc
        self.current_lr = lr
        self.peak_gpu = peak_gpu

        self.progress.update(self.task_examples, completed=examples_seen)
        self.progress.update(self.task_steps, completed=step)

        if self.live:
            self.live.update(self._render())

    def update_eval(self, step: int, metrics: dict[str, Any]) -> None:
        self.val_acc = metrics.get("accuracy", 0.0) * 100.0
        self.val_nll = metrics.get("nll", 0.0)
        self.log(
            f"Step {step} 验证评估: 准确率={self.val_acc:.1f}%, 负对数似然(NLL)={self.val_nll:.4f}"
        )
        if self.live:
            self.live.update(self._render())

    def _render(self) -> Group:
        # 1. 顶部纯黑白标题卡片
        header_text = (
            f"Mugi Decision ({self.model_name}) 训练监控看板\n"
            f"微调: {self.method} | 题量: {self.total_examples} | 累积: {self.gradient_accumulation} | 硬件: RTX 4060 (8GB)"
        )
        header_panel = Panel(
            header_text,
            box=box.ROUNDED,
            style="bold",
            padding=(0, 2),
        )

        # 2. 状态表格 (无彩色，纯黑白加粗对比)
        loss_diff = ""
        if self.initial_loss and self.initial_loss > 0:
            pct = ((self.current_loss - self.initial_loss) / self.initial_loss) * 100
            loss_diff = f" ({pct:+.1f}%)"

        val_text = f"{self.val_acc:.1f}% (NLL:{self.val_nll:.3f})" if self.val_acc is not None else "等待首次评测..."

        metrics_table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
            expand=True,
        )
        metrics_table.add_column("训练 Loss", justify="center")
        metrics_table.add_column("Batch 准确率", justify="center")
        metrics_table.add_column("显存占用", justify="center")
        metrics_table.add_column("验证集 Acc", justify="center")

        metrics_table.add_row(
            f"{self.current_loss:.4f}{loss_diff}",
            f"{self.current_acc * 100:.1f}%",
            f"{self.peak_gpu:.2f} G / 8.0 G",
            val_text,
        )

        # 3. 底部动态日志 (纯黑白时间戳)
        log_content = "\n".join(self.logs) if self.logs else "暂无事件"
        log_panel = Panel(
            log_content,
            title="最新动态",
            title_align="left",
            box=box.ROUNDED,
            style="dim",
            padding=(0, 1),
        )

        return Group(header_panel, self.progress, metrics_table, log_panel)