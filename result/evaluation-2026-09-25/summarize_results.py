"""汇总四个 checkpoint 的同题评测，并生成可复核的报告与图。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
NAMES = ["20k_step1250", "20k_step2500", "80k_step5000", "80k_step6500"]
LABELS = ["2万题\n1250步", "2万题\n2500步", "8万题\n5000步", "8万题\n6500步"]


def load_predictions(name: str, set_name: str) -> list[dict]:
    run = ROOT / ("run_1250" if name == "20k_step1250" and (ROOT / "run_1250").exists() else "run")
    path = run / f"{name}_{set_name}_predictions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_metrics(name: str) -> dict:
    run = ROOT / ("run_1250" if name == "20k_step1250" and (ROOT / "run_1250").exists() else "run")
    return json.loads((run / "metrics.json").read_text(encoding="utf-8"))[name]


def nll(item: dict) -> float:
    return -math.log(max(item["probabilities"][item["gold_position"]], 1e-12))


def compare(before: list[dict], after: list[dict]) -> dict:
    assert len(before) == len(after)
    assert [x["id"] for x in before] == [x["id"] for x in after]
    better = sum(not a["correct"] and b["correct"] for a, b in zip(before, after))
    worse = sum(a["correct"] and not b["correct"] for a, b in zip(before, after))
    changes = sorted(
        ({"id": a["id"], "source_dataset": a["source_dataset"],
          "nll_before": nll(a), "nll_after": nll(b),
          "delta_nll": nll(b) - nll(a),
          "before_correct": a["correct"], "after_correct": b["correct"]}
         for a, b in zip(before, after)),
        key=lambda row: row["delta_nll"], reverse=True)
    return {"n": len(before), "newly_correct": better, "newly_wrong": worse,
            "mean_delta_nll": sum(row["delta_nll"] for row in changes) / len(changes),
            "largest_nll_increases": changes[:10]}


def plot(metrics: dict) -> None:
    font_path = Path("/mnt/c/Windows/Fonts/simhei.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    x = range(4)
    acc = [metrics[name]["heldout_200"]["overall"]["accuracy"] * 100 for name in NAMES]
    nll_values = [metrics[name]["heldout_200"]["overall"]["nll"] for name in NAMES]
    axes[0].plot(x, acc, "o-", color="#2171b5", linewidth=2)
    axes[0].set_ylabel("留出题准确率（%）")
    axes[0].set_ylim(min(acc) - 2, max(acc) + 2)
    axes[1].plot(x, nll_values, "o-", color="#d95f0e", linewidth=2)
    axes[1].set_ylabel("留出题 NLL")
    axes[1].set_xticks(list(x), LABELS)
    for ax, values, fmt in [(axes[0], acc, "{:.1f}"), (axes[1], nll_values, "{:.3f}")]:
        ax.grid(alpha=0.22)
        for pos, value in zip(x, values):
            ax.annotate(fmt.format(value), (pos, value), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_title("四个 checkpoint：固定 200 道留出题")
    fig.savefig(ROOT / "checkpoint_comparison.png")
    plt.close(fig)


def main() -> None:
    metrics = {name: load_metrics(name) for name in NAMES}
    predictions = {name: {set_name: load_predictions(name, set_name)
                          for set_name in ("heldout_200", "custom_20")}
                   for name in NAMES}
    for set_name, count in (("heldout_200", 200), ("custom_20", 20)):
        expected = [item["id"] for item in predictions[NAMES[0]][set_name]]
        assert len(expected) == count and len(set(expected)) == count
        for name in NAMES[1:]:
            actual = predictions[name][set_name]
            assert [item["id"] for item in actual] == expected
            assert all(a["gold_position"] == b["gold_position"]
                       for a, b in zip(predictions[NAMES[0]][set_name], actual))

    paired = {}
    for set_name in ("heldout_200", "custom_20"):
        paired[set_name] = {}
        for before_name, after_name in ((NAMES[0], NAMES[1]), (NAMES[1], NAMES[2]),
                                         (NAMES[2], NAMES[3])):
            key = f"{before_name} -> {after_name}"
            paired[set_name][key] = compare(predictions[before_name][set_name],
                                            predictions[after_name][set_name])

    summary = {"metrics": metrics, "paired": paired}
    (ROOT / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(metrics)

    lines = ["# 四个 checkpoint 的固定测试集评测", "",
             "评测日期：2026-09-25。全部模型使用同一批 200 道留出题和 20 道自拟题。",
             "留出题从五个原始数据源的 validation 分片按固定种子抽取，每源 40 题；已排除 8 万题训练集及原有 1,000 题验证集的题目指纹。",
             "ARC 的 40 题由 Easy 和 Challenge 各 20 题组成。抽样方法与哈希见 `make_test_sets.py`、`test_set_manifest.json`。",
             "推理使用训练时的候选分支提示、1024 token 截断、LoRA 和评分头；不使用语言模型自由生成。", "",
             "## 总体结果", "",
             "| checkpoint | 留出题正确数 / 200 | 留出题 NLL | 留出题 Brier | 自拟题正确数 / 20 | 自拟题 NLL |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        held = metrics[name]["heldout_200"]["overall"]
        custom = metrics[name]["custom_20"]["overall"]
        lines.append(f"| {name} | {round(held['accuracy']*200)} / 200 | {held['nll']:.4f} | "
                     f"{held['brier']:.4f} | {round(custom['accuracy']*20)} / 20 | {custom['nll']:.4f} |")

    lines += ["", "## 各来源正确数", "",
              "| checkpoint | BoolQ / 40 | CommonsenseQA / 40 | ARC / 40 | C3 / 40 | OCNLI / 40 |",
              "|---|---:|---:|---:|---:|---:|"]
    sources = ["boolq", "commonsense_qa", "ai2_arc", "c3", "ocnli"]
    for name in NAMES:
        grouped = metrics[name]["heldout_200"]["source_dataset"]
        values = [f"{round(grouped[source]['accuracy']*40)}" for source in sources]
        lines.append(f"| {name} | " + " | ".join(values) + " |")

    lines += ["", "## 相邻 checkpoint 的逐题变化", "",
              "| 测试集 | 对比 | 新答对 | 新答错 | 平均 NLL 变化 |",
              "|---|---|---:|---:|---:|"]
    for set_name in ("heldout_200", "custom_20"):
        for comparison, data in paired[set_name].items():
            lines.append(f"| {set_name} | {comparison} | {data['newly_correct']} | "
                         f"{data['newly_wrong']} | {data['mean_delta_nll']:+.4f} |")

    lines += ["", "## 自拟题逐题结果", "",
              "候选项顺序由固定种子打乱，表中写出模型选中的候选文本。", "",
              "| 题号 | 题目 | 标准答案 | 1250 步 | 2500 步 | 5000 步 | 6500 步 |",
              "|---|---|---|---|---|---|---|"]
    custom_ref = predictions[NAMES[0]]["custom_20"]
    for index, reference in enumerate(custom_ref):
        answers = []
        for name in NAMES:
            item = predictions[name]["custom_20"][index]
            chosen = item["candidates"][item["predicted_position"]]
            answers.append(("✓ " if item["correct"] else "✗ ") + chosen)
        question = reference["description"].replace("|", "\\|").replace("\n", " ")
        gold = reference["candidates"][reference["gold_position"]]
        lines.append(f"| {reference['id']} | {question} | {gold} | " + " | ".join(answers) + " |")

    lines += ["", "## 解读边界", "",
              "- 200 道留出题来自五个已有训练来源的其他样本，检验同来源新题；这不是跨来源泛化测试。",
              "- 20 道自拟题由人工设计，题目较简单，主要展示实际候选打分输出，不用于精确估计总体正确率。",
              "- 第 5,000 与 6,500 步在留出题上正确数相同，后者 NLL 更高。该变化与训练日志中原有 1,000 题验证集的 NLL 走势一致。",
              "- 原始逐题概率保存在 `run/` 与 `run_1250/` 中，可检查错误集合和高置信度错误。", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("汇总完成：REPORT.md、comparison.json、checkpoint_comparison.png")


if __name__ == "__main__":
    main()
