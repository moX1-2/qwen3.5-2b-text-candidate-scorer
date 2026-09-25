"""
Prepare a balanced 20k dataset for variable-candidate scoring (Mugi Decision / JEV):
- Exactly 10,000 Chinese + 10,000 English examples (Total 20,000 for training)
- Exactly 500 Chinese + 500 English examples (Total 1,000 for validation)
- Evenly divided across 8 classical high-quality open benchmarks (2,500 train + 125 val per dataset):
  English:
    1. boolq: 2,500 train, 125 val (2 candidates: No, Yes)
    2. ai2_arc: 2,500 train, 125 val (3-4 candidates, Easy + Challenge)
    3. commonsense_qa: 2,500 train, 125 val (5 candidates)
    4. mmlu: 2,500 train, 125 val (4 candidates)
  Chinese:
    5. ocnli: 2,500 train, 125 val (3 candidates: entailment, neutral, contradiction)
    6. c3: 2,500 train, 125 val (2-4 candidates: daily dialogue / commonsense)
    7. ceval: 2,500 train, 125 val (4 candidates)
    8. cmmlu: 2,500 train, 125 val (4 candidates)
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

SEED = 20260924

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


# 1. English: BoolQ (2 candidates)
def build_boolq() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    t_rows = parquet_rows(RAW / "boolq" / "train-00000-of-00001.parquet")
    v_rows = parquet_rows(RAW / "boolq" / "validation-00000-of-00001.parquet")
    rng = random.Random(SEED + 10)
    s_train = rng.sample(t_rows, 2500)
    s_val = rng.sample(v_rows, 125)

    def to_record(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
        p = row.get("passage", "").strip()
        q = row.get("question", "").strip()
        desc = f"Passage: {p}\nQuestion: {q}?" if p else f"Question: {q}?"
        gold = 1 if row["answer"] else 0
        return make_record(
            dataset="boolq",
            language="en",
            domain="reading_comprehension",
            source_split=split,
            source_id=f"boolq_{idx}",
            question=desc,
            candidates=["No", "Yes"],
            gold_index=gold,
            task_family="boolean_judgment",
            source_role=split,
        )

    train = [to_record(r, "train", i) for i, r in enumerate(s_train)]
    val = [to_record(r, "validation", i) for i, r in enumerate(s_val)]
    return train, val


# 2. English: AI2 ARC (ARC-Easy 1500 + ARC-Challenge 1000)
def build_arc() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    easy_t = parquet_rows(RAW / "ai2_arc" / "ARC-Easy" / "train-00000-of-00001.parquet")
    easy_v = parquet_rows(RAW / "ai2_arc" / "ARC-Easy" / "validation-00000-of-00001.parquet")
    chal_t = parquet_rows(RAW / "ai2_arc" / "ARC-Challenge" / "train-00000-of-00001.parquet")
    chal_v = parquet_rows(RAW / "ai2_arc" / "ARC-Challenge" / "validation-00000-of-00001.parquet")

    rng = random.Random(SEED + 20)
    s_train = [(r, "ARC-Easy") for r in rng.sample(easy_t, 1500)] + [
        (r, "ARC-Challenge") for r in rng.sample(chal_t, 1000)
    ]
    s_val = [(r, "ARC-Easy") for r in rng.sample(easy_v, 75)] + [
        (r, "ARC-Challenge") for r in rng.sample(chal_v, 50)
    ]
    rng.shuffle(s_train)
    rng.shuffle(s_val)

    def to_record(item: tuple[dict[str, Any], str], split: str, idx: int) -> dict[str, Any]:
        row, cfg = item
        choices = row["choices"]
        texts = [t.strip() for t in choices["text"]]
        labels = [str(label).strip() for label in choices["label"]]
        gold_label = str(row["answerKey"]).strip()
        gold_index = labels.index(gold_label) if gold_label in labels else 0
        return make_record(
            dataset="ai2_arc",
            language="en",
            domain=cfg,
            source_split=split,
            source_id=f"{cfg}_{row.get('id', idx)}_{idx}",
            question=row["question"].strip(),
            candidates=texts,
            gold_index=gold_index,
            task_family="multiple_choice",
            source_role=split,
        )

    train = [to_record(item, "train", i) for i, item in enumerate(s_train)]
    val = [to_record(item, "validation", i) for i, item in enumerate(s_val)]
    return train, val


# 3. English: CommonsenseQA (5 candidates)
def build_commonsense_qa() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    t_rows = parquet_rows(RAW / "commonsense_qa" / "data" / "train-00000-of-00001.parquet")
    v_rows = parquet_rows(RAW / "commonsense_qa" / "data" / "validation-00000-of-00001.parquet")
    rng = random.Random(SEED + 30)
    s_train = rng.sample(t_rows, 2500)
    s_val = rng.sample(v_rows, 125)

    def to_record(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
        choices = row["choices"]
        texts = [t.strip() for t in choices["text"]]
        labels = [str(label).strip() for label in choices["label"]]
        gold_label = str(row["answerKey"]).strip()
        gold_index = labels.index(gold_label) if gold_label in labels else 0
        return make_record(
            dataset="commonsense_qa",
            language="en",
            domain="commonsense",
            source_split=split,
            source_id=f"{row.get('id', idx)}_{idx}",
            question=row["question"].strip(),
            candidates=texts,
            gold_index=gold_index,
            task_family="multiple_choice",
            source_role=split,
        )

    train = [to_record(r, "train", i) for i, r in enumerate(s_train)]
    val = [to_record(r, "validation", i) for i, r in enumerate(s_val)]
    return train, val


# 4. English: MMLU (4 candidates)
def build_mmlu() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = parquet_rows(RAW / "mmlu" / "test-00000-of-00001.parquet")
    rng = random.Random(SEED + 40)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    s_train = shuffled[:2500]
    s_val = shuffled[2500:2625]

    def to_record(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
        return make_record(
            dataset="mmlu",
            language="en",
            domain=row.get("subject", "general"),
            source_split=split,
            source_id=row.get("_uid", f"mmlu_{idx}"),
            question=row["question"].strip(),
            candidates=[str(c).strip() for c in row["choices"]],
            gold_index=int(row["answer"]),
            task_family="multiple_choice",
            source_role=split,
        )

    train = [to_record(r, "train", i) for i, r in enumerate(s_train)]
    val = [to_record(r, "validation", i) for i, r in enumerate(s_val)]
    return train, val


# 5. Chinese: OCNLI (3 candidates)
def build_ocnli() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    t_rows = parquet_rows(RAW / "ocnli" / "train-00000-of-00001.parquet")
    v_rows = parquet_rows(RAW / "ocnli" / "validation-00000-of-00001.parquet")
    rng = random.Random(SEED + 50)

    valid_train = [r for r in t_rows if r.get("label") in (0, 1, 2)]
    valid_val = [r for r in v_rows if r.get("label") in (0, 1, 2)]

    s_train = rng.sample(valid_train, 2500)
    s_val = rng.sample(valid_val, 125)

    candidates = ["蕴含（能够从前提推出）", "中立（无法推断关系）", "矛盾（与前提存在冲突）"]

    def to_record(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
        p = row["sentence1"].strip()
        h = row["sentence2"].strip()
        desc = f"前提：{p}\n假设：{h}\n请判断假设与前提之间的逻辑关系："
        return make_record(
            dataset="ocnli",
            language="zh",
            domain="natural_language_inference",
            source_split=split,
            source_id=f"ocnli_{idx}",
            question=desc,
            candidates=candidates,
            gold_index=int(row["label"]),
            task_family="inference_judgment",
            source_role=split,
        )

    train = [to_record(r, "train", i) for i, r in enumerate(s_train)]
    val = [to_record(r, "validation", i) for i, r in enumerate(s_val)]
    return train, val


# 6. Chinese: C3 (2-4 candidates)
def build_c3() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    t_rows = parquet_rows(RAW / "c3" / "train-00000-of-00001.parquet")
    v_rows = parquet_rows(RAW / "c3" / "validation-00000-of-00001.parquet")
    rng = random.Random(SEED + 60)

    valid_train = [r for r in t_rows if r.get("answer") in r.get("choice", [])]
    valid_val = [r for r in v_rows if r.get("answer") in r.get("choice", [])]

    s_train = rng.sample(valid_train, 2500)
    s_val = rng.sample(valid_val, 125)

    def to_record(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
        context_lines = row.get("context", [])
        context_str = "\n".join(context_lines) if isinstance(context_lines, list) else str(context_lines)
        q = row.get("question", "").strip()
        desc = f"情境与对话：\n{context_str}\n问题：{q}"
        choices = [str(c).strip() for c in row["choice"]]
        gold = choices.index(str(row["answer"]).strip())
        return make_record(
            dataset="c3",
            language="zh",
            domain="daily_dialogue",
            source_split=split,
            source_id=f"{row.get('id', 'c3')}_{idx}",
            question=desc,
            candidates=choices,
            gold_index=gold,
            task_family="multiple_choice",
            source_role=split,
        )

    train = [to_record(r, "train", i) for i, r in enumerate(s_train)]
    val = [to_record(r, "validation", i) for i, r in enumerate(s_val)]
    return train, val


# 7. Chinese: C-Eval (4 candidates)
def build_ceval() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ceval_dir = RAW / "ceval"
    val_files = sorted(ceval_dir.glob("*/val-*.parquet"))
    test_files = sorted(ceval_dir.glob("*/test-*.parquet"))

    all_val: list[dict[str, Any]] = []
    for f in val_files:
        subj = f.parent.name
        for r in parquet_rows(f):
            r["_subject"] = subj
            all_val.append(r)

    all_test: list[dict[str, Any]] = []
    for f in test_files:
        subj = f.parent.name
        for r in parquet_rows(f):
            r["_subject"] = subj
            all_test.append(r)

    rng = random.Random(SEED + 70)
    valid_test = [r for r in all_test if r.get("answer") in ("A", "B", "C", "D")]
    valid_val = [r for r in all_val if r.get("answer") in ("A", "B", "C", "D")]

    s_train = rng.sample(valid_test, 2500)
    s_val = rng.sample(valid_val, 125)

    def to_record(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        cands = [row["A"], row["B"], row["C"], row["D"]]
        subj = row["_subject"]
        return make_record(
            dataset="ceval",
            language="zh",
            domain=subj,
            source_split="test" if split == "train" else "val",
            source_id=f"{subj}_{row.get('id', idx)}_{idx}",
            question=row["question"].strip(),
            candidates=cands,
            gold_index=mapping[row["answer"].strip()],
            task_family="multiple_choice",
            source_role=split,
        )

    train = [to_record(r, "train", i) for i, r in enumerate(s_train)]
    val = [to_record(r, "validation", i) for i, r in enumerate(s_val)]
    return train, val


# 8. Chinese: CMMLU (4 candidates)
def build_cmmlu() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    zip_path = RAW / "cmmlu" / "cmmlu_v1_0_1.zip"
    all_test: list[tuple[str, dict[str, Any]]] = []
    all_dev: list[tuple[str, dict[str, Any]]] = []

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("test/") and name.endswith(".csv"):
                subj = Path(name).stem
                with z.open(name) as f:
                    reader = csv.DictReader(TextIOWrapper(f, encoding="utf-8"))
                    for row in reader:
                        all_test.append((subj, row))
            elif name.startswith("dev/") and name.endswith(".csv"):
                subj = Path(name).stem
                with z.open(name) as f:
                    reader = csv.DictReader(TextIOWrapper(f, encoding="utf-8"))
                    for row in reader:
                        all_dev.append((subj, row))

    valid_test = [item for item in all_test if item[1].get("Answer", "").strip() in ("A", "B", "C", "D")]
    valid_dev = [item for item in all_dev if item[1].get("Answer", "").strip() in ("A", "B", "C", "D")]

    rng = random.Random(SEED + 80)
    s_train = rng.sample(valid_test, 2500)
    s_val = rng.sample(valid_dev, 125)

    def to_record(item: tuple[str, dict[str, Any]], split: str, idx: int) -> dict[str, Any]:
        subj, row = item
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        cands = [row["A"], row["B"], row["C"], row["D"]]
        return make_record(
            dataset="cmmlu",
            language="zh",
            domain=subj,
            source_split="test" if split == "train" else "dev",
            source_id=f"{subj}_{row.get('id', idx)}_{idx}",
            question=row["Question"].strip(),
            candidates=cands,
            gold_index=mapping[row["Answer"].strip()],
            task_family="multiple_choice",
            source_role=split,
        )

    train = [to_record(item, "train", i) for i, item in enumerate(s_train)]
    val = [to_record(item, "validation", i) for i, item in enumerate(s_val)]
    return train, val


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    print("Gathering datasets...")
    boolq_t, boolq_v = build_boolq()
    arc_t, arc_v = build_arc()
    csqa_t, csqa_v = build_commonsense_qa()
    mmlu_t, mmlu_v = build_mmlu()

    ocnli_t, ocnli_v = build_ocnli()
    c3_t, c3_v = build_c3()
    ceval_t, ceval_v = build_ceval()
    cmmlu_t, cmmlu_v = build_cmmlu()

    all_train = (
        boolq_t + arc_t + csqa_t + mmlu_t +
        ocnli_t + c3_t + ceval_t + cmmlu_t
    )
    all_val = (
        boolq_v + arc_v + csqa_v + mmlu_v +
        ocnli_v + c3_v + ceval_v + cmmlu_v
    )

    rng = random.Random(SEED + 999)
    rng.shuffle(all_train)
    rng.shuffle(all_val)

    # Validate uniqueness
    train_ids = [r["id"] for r in all_train]
    if len(train_ids) != len(set(train_ids)):
        dup = [k for k, v in Counter(train_ids).items() if v > 1]
        raise ValueError(f"Found {len(dup)} duplicate IDs in train: {dup[:5]}")

    val_ids = [r["id"] for r in all_val]
    if len(val_ids) != len(set(val_ids)):
        dup = [k for k, v in Counter(val_ids).items() if v > 1]
        raise ValueError(f"Found {len(dup)} duplicate IDs in val: {dup[:5]}")

    write_jsonl(SELECTED / "train.jsonl", all_train)
    write_jsonl(SELECTED / "validation.jsonl", all_val)

    print("\n--- Summary ---")
    print(f"Train total: {len(all_train)}")
    print(f"Validation total: {len(all_val)}")
    print("Train by Dataset:", dict(Counter(r["source_dataset"] for r in all_train)))
    print("Train by Language:", dict(Counter(r["language"] for r in all_train)))
    print("Train by Candidate Count:", dict(sorted(Counter(len(r["candidates"]) for r in all_train).items())))


if __name__ == "__main__":
    main()