import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot():
    history_file = Path("runs/mugi-decision/history.jsonl")
    if not history_file.is_file():
        print("No history.jsonl found!")
        return

    steps, losses, accs, val_steps, val_accs = [], [], [], [], []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            steps.append(d["step"])
            losses.append(d["loss"])
            accs.append(d.get("accuracy", 0.0) * 100)
            if "val_acc" in d:
                val_steps.append(d["step"])
                val_accs.append(d["val_acc"] * 100)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # 1. Loss 曲线 (纯黑白/深灰学术风)
    ax1.plot(steps, losses, marker="o", markersize=4, color="#222222", linewidth=2.0, label="Training Loss")
    ax1.set_title("Training Loss Convergence", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Optimizer Step", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Accuracy 曲线
    ax2.plot(steps, accs, marker="s", markersize=4, color="#444444", linewidth=1.8, linestyle="--", label="Batch Acc (%)")
    if val_steps:
        ax2.plot(val_steps, val_accs, marker="^", markersize=8, color="#000000", linewidth=2.2, label=f"Val Acc (Best: {max(val_accs):.1f}%)")
    ax2.set_title("Accuracy & Validation Performance", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Optimizer Step", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(20, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    out_path = Path("runs/mugi-decision/loss_curve.png")
    plt.savefig(out_path, dpi=200)
    print(f"Generated Loss curve chart at: {out_path}")

if __name__ == "__main__":
    plot()