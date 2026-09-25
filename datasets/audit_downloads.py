"""Read-only inventory for the downloaded ARC, CommonsenseQA, C-Eval and CMMLU files.

Run from the project root with:
    uv run --no-project --with pyarrow -- python3 new/datasets/audit_downloads.py
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent / "raw"
PARQUET_DATASETS = {
    "ai2_arc": ("train", "validation", "test", "answerKey"),
    "commonsense_qa": ("train", "validation", "test", "answerKey"),
    "ceval": ("dev", "val", "test", "answer"),
}


def audit_parquet_dataset(name: str, config: tuple[str, ...]) -> None:
    train_name, validation_name, test_name, label_name = config
    files = sorted((ROOT / name).rglob("*.parquet"))
    print(f"\n[{name}] parquet_files={len(files)}")
    for split in (train_name, validation_name, test_name):
        split_files = [path for path in files if path.name.startswith(f"{split}-")]
        rows = 0
        missing = 0
        empty = 0
        invalid_labels = 0
        candidate_counts: Counter[int] = Counter()
        for path in split_files:
            table = pq.read_table(path)
            rows += table.num_rows
            labels = table.column(label_name).to_pylist()
            missing += sum(value is None for value in labels)
            empty += sum(value is not None and not str(value).strip() for value in labels)
            for row in table.to_pylist():
                if name == "ceval":
                    options = [row.get(letter) for letter in "ABCD"]
                    options = [option for option in options if option is not None and str(option).strip()]
                    valid_labels = set("ABCD")
                else:
                    choices = row.get("choices") or {}
                    options = choices.get("text") or []
                    valid_labels = set(choices.get("label") or [])
                candidate_counts[len(options)] += 1
                answer = row.get(label_name)
                if answer is not None and str(answer).strip() and answer not in valid_labels:
                    invalid_labels += 1
        print(
            f"  split={split} files={len(split_files)} rows={rows} "
            f"missing_labels={missing} empty_labels={empty} invalid_labels={invalid_labels} "
            f"candidate_counts={dict(sorted(candidate_counts.items()))}"
        )
    if files:
        print(f"  fields={pq.read_schema(files[0]).names}")


def audit_cmmlu() -> None:
    archive = ROOT / "cmmlu" / "cmmlu_v1_0_1.zip"
    with zipfile.ZipFile(archive) as source:
        csv_files = [name for name in source.namelist() if name.endswith(".csv")]
        print(f"\n[cmmlu] archive_bytes={archive.stat().st_size} csv_files={len(csv_files)}")
        for split in ("dev", "test"):
            names = [name for name in csv_files if name.startswith(f"{split}/")]
            row_count = 0
            columns: list[str] = []
            missing_labels = 0
            empty_labels = 0
            invalid_labels = 0
            candidate_counts: Counter[int] = Counter()
            for name in names:
                with source.open(name) as binary:
                    text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
                    reader = csv.DictReader(text)
                    if not columns:
                        columns = reader.fieldnames or []
                    for row in reader:
                        row_count += 1
                        answer = row.get("Answer", row.get("answer"))
                        if answer is None:
                            missing_labels += 1
                        elif not answer.strip():
                            empty_labels += 1
                        candidate_counts[sum(bool(row.get(letter, "").strip()) for letter in "ABCD")] += 1
                        if answer and answer not in set("ABCD"):
                            invalid_labels += 1
            print(
                f"  split={split} files={len(names)} rows={row_count} "
                f"missing_labels={missing_labels} empty_labels={empty_labels} "
                f"invalid_labels={invalid_labels} candidate_counts={dict(sorted(candidate_counts.items()))}"
            )
            print(f"  fields={columns}")


def main() -> None:
    for name, config in PARQUET_DATASETS.items():
        audit_parquet_dataset(name, config)
    audit_cmmlu()


if __name__ == "__main__":
    main()
