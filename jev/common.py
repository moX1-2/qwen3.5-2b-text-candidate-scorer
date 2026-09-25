"""Data validation, prompts, and metrics for the JEV candidate scorer."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NEW = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = NEW / "models" / "Qwen3.5-2B-Text"
DEFAULT_TRAIN = NEW / "datasets" / "selected" / "train.jsonl"
DEFAULT_VALIDATION = NEW / "datasets" / "selected" / "validation.jsonl"


@dataclass(frozen=True)
class Example:
    id: str
    description: str
    candidates: tuple[str, ...]
    gold_index: int
    language: str
    source_dataset: str
    domain: str


def load_examples(path: Path, limit: int | None = None) -> list[Example]:
    examples: list[Example] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw: dict[str, Any] = json.loads(line)
            identifier = str(raw.get("id", f"{path.name}:{line_number}"))
            description = raw.get("description")
            candidates = raw.get("candidates")
            gold = raw.get("gold_index")
            if identifier in seen:
                raise ValueError(f"Duplicate id in {path}: {identifier}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Empty description at {path}:{line_number}")
            if not isinstance(candidates, list) or not 2 <= len(candidates) <= len(LETTERS):
                raise ValueError(f"Invalid candidates at {path}:{line_number}")
            if any(not isinstance(value, str) or not value.strip() for value in candidates):
                raise ValueError(f"Empty candidate at {path}:{line_number}")
            if not isinstance(gold, int) or isinstance(gold, bool) or not 0 <= gold < len(candidates):
                raise ValueError(f"Invalid gold_index at {path}:{line_number}")
            if raw.get("language", "zh") not in {"zh", "en"}:
                raise ValueError(f"Unsupported language at {path}:{line_number}")
            seen.add(identifier)
            examples.append(
                Example(
                    id=identifier,
                    description=description.strip(),
                    candidates=tuple(value.strip() for value in candidates),
                    gold_index=gold,
                    language=raw.get("language", "zh"),
                    source_dataset=str(raw.get("source_dataset", "unknown")),
                    domain=str(raw.get("domain", "unknown")),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No examples in {path}")
    return examples


def ordered_indices(example: Example, order: str, seed: int) -> list[int]:
    indices = list(range(len(example.candidates)))
    if order == "original":
        return indices
    if order == "reverse":
        return indices[::-1]
    if order == "shuffle":
        digest = hashlib.sha256(f"{seed}:{example.id}".encode("utf-8")).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(indices)
        return indices
    raise ValueError(f"Unsupported candidate order: {order}")


def option_lines(example: Example, indices: list[int]) -> str:
    return "\n".join(
        f"{LETTERS[position]}. {example.candidates[original]}"
        for position, original in enumerate(indices)
    )


def zero_shot_prompt(example: Example, indices: list[int], style: str, tokenizer: Any) -> str:
    choices = option_lines(example, indices)
    if example.language == "zh":
        content = (
            "阅读问题和候选项，选出唯一正确项。"
            "只输出一个大写选项字母，不输出解释。\n"
            f"问题：{example.description}\n候选项：\n{choices}"
        )
        ending = "\n答案："
    else:
        content = (
            "Choose the single correct option. "
            "Output one uppercase option letter and no explanation.\n"
            f"Question: {example.description}\nOptions:\n{choices}"
        )
        ending = "\nAnswer:"
    if style == "plain":
        return content + ending
    if style == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    raise ValueError(f"Unsupported prompt style: {style}")


def branch_prompts(example: Example) -> list[str]:
    indices = list(range(len(example.candidates)))
    choices = option_lines(example, indices)
    prompts = []
    for position, candidate in enumerate(example.candidates):
        if example.language == "zh":
            prompt = (
                f"问题：{example.description}\n完整候选列表：\n{choices}\n"
                f"待判断候选：{LETTERS[position]}. {candidate}\n判断："
            )
        else:
            prompt = (
                f"Question: {example.description}\nAll options:\n{choices}\n"
                f"Candidate to assess: {LETTERS[position]}. {candidate}\nJudgment:"
            )
        prompts.append(prompt)
    return prompts


def encode_texts(tokenizer: Any, texts: list[str], max_length: int, example_id: str) -> Any:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=False,
        add_special_tokens=False,
        return_tensors="pt",
    )
    longest = int(batch["attention_mask"].sum(dim=1).max().item())
    if longest > max_length:
        raise ValueError(
            f"{example_id}: token length {longest} exceeds --max-length {max_length}; "
            "increase the limit or review the sample"
        )
    return batch


def metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No prediction records")
    count = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    nll = 0.0
    brier = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for record in records:
        probabilities = record["probabilities"]
        gold = record["gold_position"]
        nll -= math.log(max(probabilities[gold], 1e-12))
        brier += sum(
            (probability - (1.0 if position == gold else 0.0)) ** 2
            for position, probability in enumerate(probabilities)
        )
        confidence = max(probabilities)
        bin_index = min(int(confidence * 10), 9)
        bins[bin_index][0] += 1
        bins[bin_index][1] += confidence
        bins[bin_index][2] += float(record["correct"])
    ece = sum(
        abs((bin_data[1] / bin_data[0]) - (bin_data[2] / bin_data[0]))
        * bin_data[0]
        / count
        for bin_data in bins
        if bin_data[0]
    )
    return {
        "count": count,
        "accuracy": correct / count,
        "nll": nll / count,
        "brier": brier / count,
        "ece_10_bins": ece,
    }


def stratified_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": metrics(records)}
    for field in ("language", "source_dataset", "candidate_count"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(str(record[field]), []).append(record)
        result[field] = {name: metrics(group) for name, group in sorted(groups.items())}
    return result
