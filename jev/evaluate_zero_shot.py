"""Evaluate the unmodified text model by scoring next-token option letters."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import (
    DEFAULT_MODEL,
    DEFAULT_VALIDATION,
    LETTERS,
    encode_texts,
    load_examples,
    ordered_indices,
    stratified_metrics,
    zero_shot_prompt,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--styles", nargs="+", choices=("plain", "chat"), default=["plain"])
    parser.add_argument(
        "--orders", nargs="+", choices=("original", "reverse", "shuffle"), default=["original", "reverse"]
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_length <= 0 or (args.max_examples is not None and args.max_examples <= 0):
        parser.error("Length and example limits must be positive")
    return args


def main() -> None:
    args = arguments()
    examples = load_examples(args.data, args.max_examples)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_tokens = {}
    for letter in LETTERS[: max(len(example.candidates) for example in examples)]:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Option label {letter!r} is not a single token: {token_ids}")
        letter_tokens[letter] = token_ids[0]

    if args.dry_run:
        for example in examples[:3]:
            for style in args.styles:
                indices = ordered_indices(example, args.orders[0], args.seed)
                prompt = zero_shot_prompt(example, indices, style, tokenizer)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                print(
                    json.dumps(
                        {
                            "id": example.id,
                            "style": style,
                            "token_length": len(ids),
                            "prompt": prompt,
                            "gold_position": indices.index(example.gold_index),
                        },
                        ensure_ascii=False,
                    )
                )
        return

    import torch
    from transformers import Qwen3_5ForCausalLM

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if device == "cuda" else "fp32"
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; choose --precision fp16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    load_options = {
        "dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        load_options["device_map"] = {"": "cuda:0"}
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, **load_options)
    if device == "cpu":
        model.to(device)
    model.eval()
    model.config.use_cache = False
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    all_metrics = {}
    started = time.monotonic()
    total = len(examples) * len(args.styles) * len(args.orders)
    completed = 0
    with prediction_path.open("w", encoding="utf-8") as output, torch.inference_mode():
        for style in args.styles:
            for order in args.orders:
                condition = f"{style}/{order}"
                records = []
                for example in examples:
                    indices = ordered_indices(example, order, args.seed)
                    prompt = zero_shot_prompt(example, indices, style, tokenizer)
                    batch = encode_texts(tokenizer, [prompt], args.max_length, example.id)
                    batch = {name: tensor.to(device) for name, tensor in batch.items()}
                    logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    option_ids = torch.tensor(
                        [letter_tokens[LETTERS[position]] for position in range(len(indices))],
                        device=device,
                    )
                    option_logits = logits.index_select(0, option_ids)
                    probabilities = torch.softmax(option_logits, dim=0).cpu().tolist()
                    predicted_position = max(range(len(indices)), key=probabilities.__getitem__)
                    gold_position = indices.index(example.gold_index)
                    record = {
                        "id": example.id,
                        "condition": condition,
                        "language": example.language,
                        "source_dataset": example.source_dataset,
                        "domain": example.domain,
                        "candidate_count": len(indices),
                        "order": indices,
                        "gold_position": gold_position,
                        "predicted_position": predicted_position,
                        "predicted_original_index": indices[predicted_position],
                        "correct": predicted_position == gold_position,
                        "option_logits": option_logits.cpu().tolist(),
                        "probabilities": probabilities,
                    }
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(
                            f"{completed}/{total} examples; elapsed {time.monotonic() - started:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                all_metrics[condition] = stratified_metrics(records)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "training": False,
        "precision": precision,
        "device": device,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "conditions": all_metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
