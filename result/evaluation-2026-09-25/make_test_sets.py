"""固定抽取未参与训练和既有验证的测试题，并生成手写候选判断题。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets" / "raw"
SELECTED = ROOT / "datasets" / "selected"
OUT = Path(__file__).resolve().parent
SEED = 20260925


def fingerprint(record: dict) -> str:
    text = record["description"].strip()
    candidates = "||".join(str(x).strip() for x in record["candidates"])
    return f"{record['source_dataset']}@@{text}@@{candidates}"


def rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def record(identifier: str, source: str, language: str, domain: str,
           description: str, candidates: list[str], gold: int) -> dict:
    return {
        "id": identifier,
        "source_dataset": source,
        "source_split": "validation",
        "language": language,
        "domain": domain,
        "description": description.strip(),
        "candidates": [str(value).strip() for value in candidates],
        "gold_index": int(gold),
    }


def source_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {}

    boolq = []
    for i, row in enumerate(rows(RAW / "boolq" / "validation-00000-of-00001.parquet")):
        passage = str(row.get("passage", "")).strip()
        question = str(row.get("question", "")).strip()
        description = f"Passage: {passage}\nQuestion: {question}?" if passage else f"Question: {question}?"
        boolq.append(record(f"heldout:boolq:{i}", "boolq", "en", "reading_comprehension",
                            description, ["No", "Yes"], 1 if row["answer"] else 0))
    pools["boolq"] = boolq

    csqa = []
    for i, row in enumerate(rows(RAW / "commonsense_qa" / "data" / "validation-00000-of-00001.parquet")):
        choices = row["choices"]
        labels = [str(x).strip() for x in choices["label"]]
        answer = str(row["answerKey"]).strip()
        if answer not in labels:
            continue
        csqa.append(record(f"heldout:commonsense_qa:{i}", "commonsense_qa", "en", "commonsense",
                           row["question"], choices["text"], labels.index(answer)))
    pools["commonsense_qa"] = csqa

    for split in ("ARC-Easy", "ARC-Challenge"):
        arc = []
        path = RAW / "ai2_arc" / split / "validation-00000-of-00001.parquet"
        for i, row in enumerate(rows(path)):
            choices = row["choices"]
            labels = [str(x).strip() for x in choices["label"]]
            answer = str(row["answerKey"]).strip()
            if answer not in labels:
                continue
            arc.append(record(f"heldout:ai2_arc:{split}:{i}", "ai2_arc", "en", split,
                              row["question"], choices["text"], labels.index(answer)))
        pools[split] = arc

    ocnli = []
    options = ["蕴含（能够从前提推出）", "中立（无法推断关系）", "矛盾（与前提存在冲突）"]
    for i, row in enumerate(rows(RAW / "ocnli" / "validation-00000-of-00001.parquet")):
        if row.get("label") not in (0, 1, 2):
            continue
        description = (f"前提：{row['sentence1'].strip()}\n假设：{row['sentence2'].strip()}"
                       "\n请判断假设与前提之间的逻辑关系：")
        ocnli.append(record(f"heldout:ocnli:{i}", "ocnli", "zh", "natural_language_inference",
                            description, options, int(row["label"])))
    pools["ocnli"] = ocnli

    c3 = []
    for i, row in enumerate(rows(RAW / "c3" / "validation-00000-of-00001.parquet")):
        if row.get("answer") not in row.get("choice", []):
            continue
        context = row.get("context", [])
        context = "\n".join(context) if isinstance(context, list) else str(context)
        description = f"情境与对话：\n{context}\n问题：{row['question'].strip()}"
        choices = [str(x).strip() for x in row["choice"]]
        c3.append(record(f"heldout:c3:{i}", "c3", "zh", "daily_dialogue",
                         description, choices, choices.index(str(row["answer"]).strip())))
    pools["c3"] = c3
    return pools


CUSTOM_QUESTIONS = [
    ("zh", "arithmetic", "18 加 27 等于多少？", ["44", "45", "46"], "45"),
    ("zh", "arithmetic", "一本书 80 页，已经读了 35 页，还剩多少页？", ["35 页", "45 页", "55 页"], "45 页"),
    ("zh", "calendar", "今天是星期三，两天后是星期几？", ["星期四", "星期五", "星期六"], "星期五"),
    ("zh", "logic", "甲比乙高，乙比丙高。三人中谁最高？", ["甲", "乙", "丙"], "甲"),
    ("zh", "reading", "小王把蓝色杯子放在书架上，把红色杯子放在桌上。哪个杯子在书架上？", ["蓝色杯子", "红色杯子", "两个杯子"], "蓝色杯子"),
    ("zh", "unit", "一小时有多少分钟？", ["30 分钟", "60 分钟", "100 分钟"], "60 分钟"),
    ("zh", "arithmetic", "三张 20 元纸币一共是多少元？", ["40 元", "60 元", "80 元"], "60 元"),
    ("zh", "logic", "盒子里只有苹果和梨。取出一个水果，确认它不是苹果。它是什么？", ["苹果", "梨", "无法判断"], "梨"),
    ("zh", "reading", "通知写着会议 14:00 开始，13:45 签到。签到应在几点？", ["13:45", "14:00", "14:45"], "13:45"),
    ("zh", "arithmetic", "一件商品原价 50 元，减价 10 元后售价多少？", ["40 元", "45 元", "60 元"], "40 元"),
    ("en", "arithmetic", "What is 7 multiplied by 8?", ["54", "56", "64"], "56"),
    ("en", "calendar", "If today is Monday, what day is three days later?", ["Wednesday", "Thursday", "Friday"], "Thursday"),
    ("en", "reading", "Maya put the keys in the drawer and the wallet on the table. Where are the keys?", ["In the drawer", "On the table", "In the wallet"], "In the drawer"),
    ("en", "unit", "How many centimeters are in one meter?", ["10", "100", "1000"], "100"),
    ("en", "logic", "All squares have four sides. A shape is a square. How many sides does it have?", ["Three", "Four", "Five"], "Four"),
    ("en", "arithmetic", "A train leaves at 9:00 and arrives at 11:30. How long is the trip?", ["2 hours", "2 hours 30 minutes", "3 hours 30 minutes"], "2 hours 30 minutes"),
    ("en", "reading", "The note says the library closes at 6 p.m. When does it close?", ["5 p.m.", "6 p.m.", "7 p.m."], "6 p.m."),
    ("en", "logic", "Lena is older than Sam, and Sam is older than Kai. Who is youngest?", ["Lena", "Sam", "Kai"], "Kai"),
    ("en", "arithmetic", "A box contains 12 pencils. Four are removed. How many remain?", ["6", "8", "10"], "8"),
    ("en", "commonsense", "Which tool is normally used to cut a sheet of paper?", ["Scissors", "Spoon", "Pillow"], "Scissors"),
]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = set()
    for name in ("train_80k.jsonl", "validation.jsonl"):
        with (SELECTED / name).open(encoding="utf-8") as handle:
            excluded.update(fingerprint(json.loads(line)) for line in handle)

    rng = random.Random(SEED)
    targets = {"boolq": 40, "commonsense_qa": 40, "ARC-Easy": 20,
               "ARC-Challenge": 20, "ocnli": 40, "c3": 40}
    selected = []
    available_counts = {}
    for name, pool in source_pools().items():
        available = [item for item in pool if fingerprint(item) not in excluded]
        available_counts[name] = len(available)
        assert len(available) >= targets[name], (name, len(available))
        chosen = rng.sample(available, targets[name])
        selected.extend(chosen)
    assert len(selected) == 200
    fingerprints = [fingerprint(item) for item in selected]
    assert len(fingerprints) == len(set(fingerprints))
    assert not set(fingerprints).intersection(excluded)
    rng.shuffle(selected)
    write_jsonl(OUT / "heldout_200.jsonl", selected)

    custom = []
    for index, (language, domain, question, choices, answer) in enumerate(CUSTOM_QUESTIONS, 1):
        shuffled = list(choices)
        rng.shuffle(shuffled)
        custom.append(record(f"custom:{index:03d}", "custom", language, domain,
                             question, shuffled, shuffled.index(answer)))
        custom[-1]["source_split"] = "hand_authored"
    write_jsonl(OUT / "custom_20.jsonl", custom)

    metadata = {
        "seed": SEED,
        "excluded_fingerprint_count": len(excluded),
        "heldout_count": len(selected),
        "heldout_sources": dict(Counter(item["source_dataset"] for item in selected)),
        "available_by_pool": available_counts,
        "custom_count": len(custom),
        "sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                   for name in ("heldout_200.jsonl", "custom_20.jsonl")},
    }
    (OUT / "test_set_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
