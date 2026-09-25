"""
Generate an 80k dataset (50k Chinese, 30k English) for variable-candidate scoring:
- Exactly 50,000 Chinese + 30,000 English = 80,000 examples
- Fully preserves all 20,000 training examples from selected/train.jsonl (Data Replay)
- Strictly excludes all 1,000 examples from selected/validation.jsonl (Zero Contamination)
- Even, high-quality domain distribution across 8 classical benchmarks:
  Chinese (50,000):
    1. ocnli: 18,000 (2,500 existing + 15,500 new)
    2. c3: 11,000 (2,500 existing + 8,500 new)
    3. ceval: 10,500 (2,500 existing + 8,000 new)
    4. cmmlu: 10,500 (2,500 existing + 8,000 new)
  English (30,000):
    5. mmlu: 11,000 (2,500 existing + 8,500 new)
    6. commonsense_qa: 8,000 (2,500 existing + 5,500 new)
    7. boolq: 8,000 (2,500 existing + 5,500 new)
    8. ai2_arc: 3,000 (2,500 existing + 500 new)
"""

from __future__ import annotations

import csv
import json
import random
import zipfile
from collections import Counter
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DATASETS_DIR = Path(__file__).resolve().parent
RAW = DATASETS_DIR / "raw"
SELECTED = DATASETS_DIR / "selected"

SEED = 20260925

LICENSE_MAP = {
    "boolq": "CC BY-SA 3.0",
    "ai2_arc": "CC BY-SA 4.0",
    "commonsense_qa": "MIT",
    "mmlu": "MIT",
    "ocnli": "CC BY-SA 4.0",
    "c3": "CC BY-SA 4.0",
    "ceval": "CC BY-NC-SA 4.0",
    "cmmlu": "CC BY-NC-SA 4.0",
}

TARGETS = {
    "ocnli": 18000,
    "c3": 11000,
    "ceval": 10500,
    "cmmlu": 10500,
    "mmlu": 11000,
    "commonsense_qa": 8000,
    "boolq": 8000,
    "ai2_arc": 3000,
}


def make_record(
    *,
    dataset: str,
    language: str,
    domain: str,
    source_split: str,
    source_id: str,
    question: str,
    candidates: list[str],
    gold_index: int,
    task_family: str,
    source_role: str,
) -> dict[str, Any]:
    cleaned_candidates = [str(c).strip() for c in candidates]
    identifier = f"{dataset}:{domain}:{source_split}:{source_id}"
    return {
        "id": identifier,
        "description": question.strip(),
        "candidates": cleaned_candidates,
        "gold_index": gold_index,
        "language": language,
        "task_family": task_family,
        "domain": domain,
        "source_dataset": dataset,
        "source_id": str(source_id),
        "source_split": source_split,
        "source_role": source_role,
        "license": LICENSE_MAP[dataset],
    }


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    stem = path.stem.replace("-00000-of-00001", "")
    parent = path.parent.name
    rows = []
    for i, r in enumerate(table.to_pylist()):
        r["_uid"] = f"{parent}_{stem}_{i}"
        rows.append(r)
    return rows


def record_fingerprint(rec: dict[str, Any]) -> str:
    desc = rec["description"].strip()
    cands = "||".join(c.strip() for c in rec["candidates"])
    return f"{rec['source_dataset']}@@{desc}@@{cands}"


def build_all_boolq() -> list[dict[str, Any]]:
    t_rows = parquet_rows(RAW / "boolq" / "train-00000-of-00001.parquet")
    records = []
    for i, row in enumerate(t_rows):
        p = row.get("passage", "").strip()
        q = row.get("question", "").strip()
        desc = f"Passage: {p}\nQuestion: {q}?" if p else f"Question: {q}?"
        gold = 1 if row["answer"] else 0
        records.append(
            make_record(
                dataset="boolq",
                language="en",
                domain="reading_comprehension",
                source_split="train",
                source_id=f"boolq_{i}",
                question=desc,
                candidates=["No", "Yes"],
                gold_index=gold,
                task_family="boolean_judgment",
                source_role="train",
            )
        )
    return records


def build_all_arc() -> list[dict[str, Any]]:
    easy_t = parquet_rows(RAW / "ai2_arc" / "ARC-Easy" / "train-00000-of-00001.parquet")
    chal_t = parquet_rows(RAW / "ai2_arc" / "ARC-Challenge" / "train-00000-of-00001.parquet")
    records = []
    for cfg, rows in [("ARC-Easy", easy_t), ("ARC-Challenge", chal_t)]:
        for i, row in enumerate(rows):
            choices = row["choices"]
            texts = [t.strip() for t in choices["text"]]
            labels = [str(label).strip() for label in choices["label"]]
            gold_label = str(row["answerKey"]).strip()
            gold_index = labels.index(gold_label) if gold_label in labels else 0
            records.append(
                make_record(
                    dataset="ai2_arc",
                    language="en",
                    domain=cfg,
                    source_split="train",
                    source_id=f"{cfg}_{row.get('id', i)}_{i}",
                    question=row["question"].strip(),
                    candidates=texts,
                    gold_index=gold_index,
                    task_family="multiple_choice",
                    source_role="train",
                )
            )
    return records


def build_all_commonsense_qa() -> list[dict[str, Any]]:
    t_rows = parquet_rows(RAW / "commonsense_qa" / "data" / "train-00000-of-00001.parquet")
    records = []
    for i, row in enumerate(t_rows):
        choices = row["choices"]
        texts = [t.strip() for t in choices["text"]]
        labels = [str(label).strip() for label in choices["label"]]
        gold_label = str(row["answerKey"]).strip()
        gold_index = labels.index(gold_label) if gold_label in labels else 0
        records.append(
            make_record(
                dataset="commonsense_qa",
                language="en",
                domain="commonsense",
                source_split="train",
                source_id=f"{row.get('id', i)}_{i}",
                question=row["question"].strip(),
                candidates=texts,
                gold_index=gold_index,
                task_family="multiple_choice",
                source_role="train",
            )
        )
    return records


def build_all_mmlu() -> list[dict[str, Any]]:
    rows = parquet_rows(RAW / "mmlu" / "test-00000-of-00001.parquet")
    records = []
    for i, row in enumerate(rows):
        records.append(
            make_record(
                dataset="mmlu",
                language="en",
                domain=row.get("subject", "general"),
                source_split="test",
                source_id=row.get("_uid", f"mmlu_{i}"),
                question=row["question"].strip(),
                candidates=[str(c).strip() for c in row["choices"]],
                gold_index=int(row["answer"]),
                task_family="multiple_choice",
                source_role="train",
            )
        )
    return records


def build_all_ocnli() -> list[dict[str, Any]]:
    t_rows = parquet_rows(RAW / "ocnli" / "train-00000-of-00001.parquet")
    candidates = ["蕴含（能够从前提推出）", "中立（无法推断关系）", "矛盾（与前提存在冲突）"]
    records = []
    for i, row in enumerate(t_rows):
        if row.get("label") not in (0, 1, 2):
            continue
        p = row["sentence1"].strip()
        h = row["sentence2"].strip()
        desc = f"前提：{p}\n假设：{h}\n请判断假设与前提之间的逻辑关系："
        records.append(
            make_record(
                dataset="ocnli",
                language="zh",
                domain="natural_language_inference",
                source_split="train",
                source_id=f"ocnli_{i}",
                question=desc,
                candidates=candidates,
                gold_index=int(row["label"]),
                task_family="inference_judgment",
                source_role="train",
            )
        )
    return records


def build_all_c3() -> list[dict[str, Any]]:
    t_rows = parquet_rows(RAW / "c3" / "train-00000-of-00001.parquet")
    records = []
    for i, row in enumerate(t_rows):
        if row.get("answer") not in row.get("choice", []):
            continue
        context_lines = row.get("context", [])
        context_str = "\n".join(context_lines) if isinstance(context_lines, list) else str(context_lines)
        q = row.get("question", "").strip()
        desc = f"情境与对话：\n{context_str}\n问题：{q}"
        choices = [str(c).strip() for c in row["choice"]]
        gold = choices.index(str(row["answer"]).strip())
        records.append(
            make_record(
                dataset="c3",
                language="zh",
                domain="daily_dialogue",
                source_split="train",
                source_id=f"{row.get('id', 'c3')}_{i}",
                question=desc,
                candidates=choices,
                gold_index=gold,
                task_family="multiple_choice",
                source_role="train",
            )
        )
    return records


def build_all_ceval() -> list[dict[str, Any]]:
    ceval_dir = RAW / "ceval"
    test_files = sorted(ceval_dir.glob("*/test-*.parquet"))
    records = []
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    for f in test_files:
        subj = f.parent.name
        for i, row in enumerate(parquet_rows(f)):
            if row.get("answer") not in mapping:
                continue
            cands = [row["A"], row["B"], row["C"], row["D"]]
            records.append(
                make_record(
                    dataset="ceval",
                    language="zh",
                    domain=subj,
                    source_split="test",
                    source_id=f"{subj}_{row.get('id', i)}_{i}",
                    question=row["question"].strip(),
                    candidates=cands,
                    gold_index=mapping[row["answer"].strip()],
                    task_family="multiple_choice",
                    source_role="train",
                )
            )
    return records


def build_all_cmmlu() -> list[dict[str, Any]]:
    zip_path = RAW / "cmmlu" / "cmmlu_v1_0_1.zip"
    records = []
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("test/") and name.endswith(".csv"):
                subj = Path(name).stem
                with z.open(name) as f:
                    reader = csv.DictReader(TextIOWrapper(f, encoding="utf-8"))
                    for i, row in enumerate(reader):
                        ans = row.get("Answer", "").strip()
                        if ans not in mapping:
                            continue
                        cands = [row["A"], row["B"], row["C"], row["D"]]
                        records.append(
                            make_record(
                                dataset="cmmlu",
                                language="zh",
                                domain=subj,
                                source_split="test",
                                source_id=f"{subj}_{row.get('id', i)}_{i}",
                                question=row["Question"].strip(),
                                candidates=cands,
                                gold_index=mapping[ans],
                                task_family="multiple_choice",
                                source_role="train",
                            )
                        )
    return records


BUILDERS = {
    "boolq": build_all_boolq,
    "ai2_arc": build_all_arc,
    "commonsense_qa": build_all_commonsense_qa,
    "mmlu": build_all_mmlu,
    "ocnli": build_all_ocnli,
    "c3": build_all_c3,
    "ceval": build_all_ceval,
    "cmmlu": build_all_cmmlu,
}


def main() -> None:
    print("=== 正在准备 80k 高质量平衡训练集 ===")

    # 1. 载入现有的 validation.jsonl 作为零泄漏排除集
    val_path = SELECTED / "validation.jsonl"
    with open(val_path, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]
    val_fps = {record_fingerprint(r) for r in val_records}
    print(f"载入现有验证集: {len(val_records)} 条，指纹数: {len(val_fps)} (严格防泄漏)")

    # 2. 载入现有的 train.jsonl (20k)，实现完整数据回放 (Data Replay)
    train_path = SELECTED / "train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        existing_train = [json.loads(line) for line in f]
    print(f"载入原 20k 训练集: {len(existing_train)} 条 (100% 完整继承保留)")

    existing_by_dataset: dict[str, list[dict[str, Any]]] = {k: [] for k in TARGETS}
    for r in existing_train:
        existing_by_dataset[r["source_dataset"]].append(r)

    train_fps = {record_fingerprint(r) for r in existing_train}

    # 3. 逐个数据集构建并抽取增量样本
    final_80k: list[dict[str, Any]] = []
    rng = random.Random(SEED)

    for ds_name, target_total in TARGETS.items():
        base_records = existing_by_dataset[ds_name]
        needed_new = target_total - len(base_records)
        print(f"\n处理数据集 [{ds_name}]: 目标总数={target_total}, 已有={len(base_records)}, 需增补={needed_new}")

        builder = BUILDERS[ds_name]
        raw_candidates = builder()

        # 严格过滤: 不能在验证集里，也不能在已有训练集里
        available_pool = []
        for r in raw_candidates:
            fp = record_fingerprint(r)
            if fp not in val_fps and fp not in train_fps:
                available_pool.append(r)
                # 放入 train_fps 防止 raw 自身有重复
                train_fps.add(fp)

        print(f"  可用新增候选池大小: {len(available_pool)}")
        if len(available_pool) < needed_new:
            raise ValueError(f"数据集 {ds_name} 候选池不足! 需要 {needed_new} 条，仅有 {len(available_pool)} 条")

        # 随机抽取所需数量，并赋予唯一 ID 避免与原有 20k 序号冲突
        sampled_new = rng.sample(available_pool, needed_new)
        for idx, item in enumerate(sampled_new, start=len(base_records)):
            item["source_id"] = f"{ds_name}_aug_{idx}"
            item["id"] = f"{item['source_dataset']}:{item['domain']}:{item['source_split']}:{item['source_id']}"
        ds_combined = base_records + sampled_new
        final_80k.extend(ds_combined)
        print(f"  [完成] 该数据集总计收录: {len(ds_combined)} 条")

    # 4. 全局 Shuffle
    rng.shuffle(final_80k)

    # 5. 最终校验
    print("\n=== 数据审计与校验 ===")
    print(f"总样本数: {len(final_80k)}")
    assert len(final_80k) == 80000, f"总数不为 80,000! 实际为 {len(final_80k)}"

    lang_counts = Counter(r["language"] for r in final_80k)
    print("语言分布:", dict(lang_counts))
    assert lang_counts["zh"] == 50000, f"中文不是 50,000! 实际为 {lang_counts['zh']}"
    assert lang_counts["en"] == 30000, f"英文不是 30,000! 实际为 {lang_counts['en']}"

    ds_counts = Counter(r["source_dataset"] for r in final_80k)
    print("数据集分布:", dict(ds_counts))

    cand_counts = Counter(len(r["candidates"]) for r in final_80k)
    print("候选项数分布:", dict(sorted(cand_counts.items())))

    # 全局 ID 唯一性严格校验
    all_ids = [r["id"] for r in final_80k]
    assert len(all_ids) == len(set(all_ids)), f"存在重复 ID! 独立 ID 数={len(set(all_ids))}, 总数={len(all_ids)}"
    print(f"全局 ID 唯一性校验通过: 独立 ID 数={len(set(all_ids))} / 80000")

    # 验证集零交叉验证
    final_fps = {record_fingerprint(r) for r in final_80k}
    intersection = final_fps.intersection(val_fps)
    print(f"与验证集重叠题目数: {len(intersection)} (必须为 0)")
    assert len(intersection) == 0, "严重错误: 训练集与验证集存在重叠数据!"

    # 写入目标文件
    out_path = SELECTED / "train_80k.jsonl"
    print(f"\n正在写入目标文件: {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final_80k:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"80k 数据集生成成功! 文件大小: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
