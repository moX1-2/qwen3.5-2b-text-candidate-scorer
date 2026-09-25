"""从训练日志重建并绘制两个阶段的指标曲线。"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
STEP_RE = re.compile(r"Step\s+(\d+)/(\d+).*?Loss:\s*([\d.]+).*?Acc:\s*([\d.]+)%")
EVAL_RE = re.compile(r"\[EVAL\] Step (\d+) .*?Acc=([\d.]+)%, NLL=([\d.]+)")


def parse_log(path: Path, *, final_80k_run: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    content = path.read_text(encoding="utf-8")
    if final_80k_run:
        marker = "[2026-09-25 03:17:51] A100 专享算力集群训练启动"
        assert marker in content, "找不到最终 8 万题训练的启动标记"
        content = content[content.index(marker) :]

    steps = {}
    evals = {}
    for line in content.splitlines():
        match = STEP_RE.search(line)
        if match:
            step, total, loss, acc = match.groups()
            steps[int(step)] = {"step": int(step), "total_steps": int(total),
                                "train_loss": float(loss), "train_acc": float(acc)}
        match = EVAL_RE.search(line)
        if match:
            step, acc, nll = match.groups()
            evals[int(step)] = {"step": int(step), "val_acc": float(acc), "val_nll": float(nll)}
    train = pd.DataFrame(sorted(steps.values(), key=lambda row: row["step"]))
    val = pd.DataFrame(sorted(evals.values(), key=lambda row: row["step"]))
    assert not train.empty and not val.empty
    return train, val


def setup() -> None:
    font_path = Path("/mnt/c/Windows/Fonts/simhei.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
    })


def annotate_epoch(ax: plt.Axes, boundary: int) -> None:
    ax.axvline(boundary, color="#555555", linestyle="--", linewidth=1.2)
    ax.text(boundary, 0.97, "首轮结束", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=9, color="#555555")


def plot_validation(val: pd.DataFrame, *, title: str, boundary: int, filename: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(val.step, val.val_acc, "o-", color="#2171b5", linewidth=1.8, markersize=4)
    axes[0].set_ylabel("验证准确率（%）")
    axes[0].set_title(title)
    axes[1].plot(val.step, val.val_nll, "o-", color="#d95f0e", linewidth=1.8, markersize=4)
    axes[1].set_ylabel("验证 NLL")
    axes[1].set_xlabel("训练步数")
    for ax in axes:
        annotate_epoch(ax, boundary)
    best = val.loc[val.val_nll.idxmin()]
    axes[1].scatter([best.step], [best.val_nll], s=80, marker="*", color="#b30000", zorder=5)
    axes[1].annotate(f"最低 NLL：{best.val_nll:.4f}（{int(best.step)} 步）",
                     (best.step, best.val_nll), xytext=(8, 12),
                     textcoords="offset points", fontsize=9)
    fig.savefig(OUT / filename)
    plt.close(fig)


def plot_training(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for ax, field, color, ylabel in [
        (axes[0], "train_loss", "#d95f0e", "训练损失"),
        (axes[1], "train_acc", "#2171b5", "训练准确率（%）"),
    ]:
        ax.plot(train.step, train[field], color=color, alpha=0.25,
                linewidth=0.8, label="日志记录点（每 50 步）")
        smooth = train[field].rolling(11, center=True, min_periods=5).mean()
        ax.plot(train.step, smooth, color=color, linewidth=2.0,
                label="11 点移动平均")
        ax.set_ylabel(ylabel)
        annotate_epoch(ax, 5000)
        ax.legend(loc="upper right", frameon=False)
    axes[0].set_title("8 万题阶段：训练指标（每个日志点对应 16 题）")
    axes[1].set_xlabel("训练步数")
    fig.savefig(OUT / "80k_训练曲线.png")
    plt.close(fig)


def main() -> None:
    setup()
    train_20k, val_20k = parse_log(ROOT / "train.log")
    train_80k, val_80k = parse_log(ROOT / "train (1).log", final_80k_run=True)

    for name, frame in [
        ("20k_训练记录.csv", train_20k),
        ("20k_验证记录.csv", val_20k),
        ("80k_训练记录.csv", train_80k),
        ("80k_验证记录.csv", val_80k),
    ]:
        frame.to_csv(OUT / name, index=False, quoting=csv.QUOTE_MINIMAL)

    plot_validation(val_80k, title="8 万题阶段：固定 1,000 题验证集", boundary=5000,
                    filename="80k_验证曲线.png")
    plot_training(train_80k)
    plot_validation(val_20k, title="2 万题阶段：固定 1,000 题验证集", boundary=1250,
                    filename="20k_验证曲线.png")
    print(f"20k：{len(train_20k)} 个训练点，{len(val_20k)} 个验证点")
    print(f"80k：{len(train_80k)} 个训练点，{len(val_80k)} 个验证点")
    print(f"图片和 CSV 已保存到 {OUT}")


if __name__ == "__main__":
    main()
