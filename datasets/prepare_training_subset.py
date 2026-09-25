"""Create a small, reproducible English/Chinese multiple-choice pilot set.

Run from the project root with:
    uv run --no-project --with pyarrow -- python3 new/datasets/prepare_training_subset.py
"""

from __future__ import annotations

import csv
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUTPUT = HERE / "selected" / "train.jsonl"
VALIDATION_OUTPUT = HERE / "selected" / "validation.jsonl"
SEED = 20260924

LICENSES = {
    "ai2_arc": "cc-by-sa-4.0",
    "commonsense_qa": "mit",
    "ceval": "cc-by-nc-sa-4.0",
    "cmmlu": "cc-by-nc-4.0",
}


def record(
    *,
    dataset: str,
    language: str,
    domain: str,
    source_split: str,
    source_id: Any,
    question: str,
    candidates: list[str],
    labels: list[str],
    answer: str,
    source_role: str,
) -> dict[str, Any]:
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported language: {language}")
    if answer not in labels:
        raise ValueError(f"Answer {answer!r} not in option labels {labels!r}")
    if len(candidates) != len(labels) or len(candidates) < 2:
        raise ValueError(f"Invalid candidate set: {dataset} {source_id}")
    return {
        "id": f"{dataset}:{domain}:{source_split}:{source_id}",
        "description": question,
        "candidates": candidates,
        "gold_index": labels.index(answer),
        "language": language,
        "task_family": "multiple_choice",
        "domain": domain,
        "source_dataset": dataset,
        "source_id": str(source_id),
        "source_split": source_split,
        "source_role": source_role,
        "license": LICENSES[dataset],
    }


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def arc_records() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    result = []
    for config in ("ARC-Easy", "ARC-Challenge"):
        path = RAW / "ai2_arc" / config / "train-00000-of-00001.parquet"
        rows = parquet_rows(path)
        for row in rng.sample(rows, 250):
            choices = row["choices"]
            result.append(
                record(
                    dataset="ai2_arc",
                    language="en",
                    domain=config,
                    source_split="train",
                    source_id=row["id"],
                    question=row["question"],
                    candidates=choices["text"],
                    labels=choices["label"],
                    answer=row["answerKey"],
                    source_role="train",
                )
            )
    return result


def commonsense_records() -> list[dict[str, Any]]:
    path = RAW / "commonsense_qa" / "data" / "train-00000-of-00001.parquet"
    rows = parquet_rows(path)
    selected = random.Random(SEED + 1).sample(rows, 500)
    result = []
    for row in selected:
        choices = row["choices"]
        result.append(
            record(
                dataset="commonsense_qa",
                language="en",
                domain="commonsense",
                source_split="train",
                source_id=row["id"],
                question=row["question"],
                candidates=choices["text"],
                labels=choices["label"],
                answer=row["answerKey"],
                source_role="train",
            )
        )
    return result


def ceval_records() -> list[dict[str, Any]]:
    result = []
    for path in sorted((RAW / "ceval").glob("*/*dev-*.parquet")):
        domain = path.parent.name
        for row in parquet_rows(path):
            labels = list("ABCD")
            result.append(
                record(
                    dataset="ceval",
                    language="zh",
                    domain=domain,
                    source_split="dev",
                    source_id=row["id"],
                    question=row["question"],
                    candidates=[row[label] for label in labels],
                    labels=labels,
                    answer=row["answer"],
                    source_role="fewshot_example_used_for_training",
                )
            )
    return result


def cmmlu_records() -> list[dict[str, Any]]:
    result = []
    archive = RAW / "cmmlu" / "cmmlu_v1_0_1.zip"
    labels = list("ABCD")
    with zipfile.ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            if not name.startswith("dev/") or not name.endswith(".csv"):
                continue
            domain = Path(name).stem
            with source.open(name) as binary:
                text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
                for row in csv.DictReader(text):
                    result.append(
                        record(
                            dataset="cmmlu",
                            language="zh",
                            domain=domain,
                            source_split="dev",
                            source_id=row.get("") or row.get("id") or "",
                            question=row["Question"],
                            candidates=[row[label] for label in labels],
                            labels=labels,
                            answer=row["Answer"].strip(),
                            source_role="fewshot_example_used_for_training",
                        )
                    )
    return result


def validation_records() -> list[dict[str, Any]]:
    result = []
    rng = random.Random(SEED + 2)
    for config in ("ARC-Easy", "ARC-Challenge"):
        path = RAW / "ai2_arc" / config / "validation-00000-of-00001.parquet"
        for row in rng.sample(parquet_rows(path), 100):
            choices = row["choices"]
            result.append(
                record(
                    dataset="ai2_arc",
                    language="en",
                    domain=config,
                    source_split="validation",
                    source_id=row["id"],
                    question=row["question"],
                    candidates=choices["text"],
                    labels=choices["label"],
                    answer=row["answerKey"],
                    source_role="validation",
                )
            )

    csqa_path = RAW / "commonsense_qa" / "data" / "validation-00000-of-00001.parquet"
    for row in rng.sample(parquet_rows(csqa_path), 200):
        choices = row["choices"]
        result.append(
            record(
                dataset="commonsense_qa",
                language="en",
                domain="commonsense",
                source_split="validation",
                source_id=row["id"],
                question=row["question"],
                candidates=choices["text"],
                labels=choices["label"],
                answer=row["answerKey"],
                source_role="validation",
            )
        )

    for path in sorted((RAW / "ceval").glob("*/*val-*.parquet")):
        domain = path.parent.name
        for row in rng.sample(parquet_rows(path), 5):
            labels = list("ABCD")
            result.append(
                record(
                    dataset="ceval",
                    language="zh",
                    domain=domain,
                    source_split="val",
                    source_id=row["id"],
                    question=row["question"],
                    candidates=[row[label] for label in labels],
                    labels=labels,
                    answer=row["answer"],
                    source_role="validation",
                )
            )
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = arc_records() + commonsense_records() + ceval_records() + cmmlu_records()
    rows.sort(key=lambda item: (item["source_dataset"], item["domain"], item["id"]))
    validation = validation_records()
    validation.sort(key=lambda item: (item["source_dataset"], item["domain"], item["id"]))
    train_ids = [row["id"] for row in rows]
    validation_ids = [row["id"] for row in validation]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Training sample IDs are not unique")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Validation sample IDs are not unique")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Training and validation IDs overlap")
    if any(row["language"] not in {"en", "zh"} for row in rows + validation):
        raise ValueError("Unexpected language in prepared data")
    write_jsonl(OUTPUT, rows)
    write_jsonl(VALIDATION_OUTPUT, validation)
    counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    for row in rows:
        counts[row["source_dataset"]] = counts.get(row["source_dataset"], 0) + 1
        language_counts[row["language"]] = language_counts.get(row["language"], 0) + 1
    validation_counts: dict[str, int] = {}
    for row in validation:
        validation_counts[row["source_dataset"]] = validation_counts.get(row["source_dataset"], 0) + 1
    print(f"train={OUTPUT.relative_to(HERE.parent)} rows={len(rows)}")
    print(f"by_dataset={counts}")
    print(f"by_language={language_counts}")
    print(f"validation={VALIDATION_OUTPUT.relative_to(HERE.parent)} rows={len(validation)} by_dataset={validation_counts}")


if __name__ == "__main__":
    main()
