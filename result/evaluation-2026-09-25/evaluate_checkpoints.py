"""在固定候选题上评测三个 LoRA 与 Score Head checkpoint。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jev"))
from common import branch_prompts, load_examples, stratified_metrics  # noqa: E402
from train_cluster import ScoreHead, prepare_examples  # noqa: E402


CHECKPOINTS = {
    "20k_step1250": ROOT / "result" / "unpacked" / "epoch-00-step-001250",
    "20k_step2500": ROOT / "result" / "unpacked" / "epoch-01-step-002500",
    "80k_step5000": ROOT / "result" / "unpacked" / "epoch-00-step-005000",
    "80k_step6500": ROOT / "result" / "unpacked" / "epoch-01-step-006500",
}
SETS = {
    "heldout_200": Path(__file__).resolve().parent / "heldout_200.jsonl",
    "custom_20": Path(__file__).resolve().parent / "custom_20.jsonl",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "run")
    parser.add_argument("--limit", type=int, default=None,
                        help="每个测试集仅评测前 N 题，用于运行前检查")
    parser.add_argument("--candidate-batch-size", type=int, default=2)
    parser.add_argument("--checkpoint", choices=list(CHECKPOINTS) + ["all"], default="all")
    args = parser.parse_args()
    if args.candidate_batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("批量大小和样本上限必须为正数")
    return args


def score_candidates(backbone: PeftModel, head: ScoreHead,
                     candidate_ids: list[torch.Tensor], batch_size: int,
                     pad_token_id: int, device: torch.device) -> list[float]:
    scores = []
    for offset in range(0, len(candidate_ids), batch_size):
        chunk = candidate_ids[offset:offset + batch_size]
        padded = torch.nn.utils.rnn.pad_sequence(
            chunk, batch_first=True, padding_value=pad_token_id).to(device)
        attention_mask = (padded != pad_token_id).long()
        hidden = backbone(input_ids=padded, attention_mask=attention_mask,
                          use_cache=False).last_hidden_state
        last_positions = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[torch.arange(hidden.shape[0], device=device), last_positions]
        scores.extend(head(last_hidden.float()).squeeze(-1).float().cpu().tolist())
        del padded, attention_mask, hidden, last_hidden
    return scores


def evaluate_one_set(backbone: PeftModel, head: ScoreHead, prepared: list,
                     batch_size: int, pad_token_id: int, device: torch.device,
                     output_path: Path) -> dict:
    records = []
    started = time.monotonic()
    with output_path.open("w", encoding="utf-8") as output, torch.inference_mode():
        for index, (example, candidate_ids) in enumerate(prepared, 1):
            scores = score_candidates(backbone, head, candidate_ids, batch_size,
                                      pad_token_id, device)
            probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).tolist()
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
            item = {
                "id": example.id,
                "language": example.language,
                "source_dataset": example.source_dataset,
                "domain": example.domain,
                "candidate_count": len(example.candidates),
                "description": example.description,
                "candidates": list(example.candidates),
                "gold_position": example.gold_index,
                "predicted_position": predicted,
                "correct": predicted == example.gold_index,
                "scores": scores,
                "probabilities": probabilities,
            }
            records.append(item)
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
            if index % 20 == 0 or index == len(prepared):
                output.flush()
                print(f"  {output_path.stem}: {index}/{len(prepared)} 题，耗时 {time.monotonic()-started:.1f} 秒", flush=True)
    metrics = stratified_metrics(records)
    metrics["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return metrics


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_names = list(CHECKPOINTS) if args.checkpoint == "all" else [args.checkpoint]
    for name in checkpoint_names:
        for required in ("adapter/adapter_model.safetensors", "score_head.pt", "state.json"):
            assert (CHECKPOINTS[name] / required).is_file(), (name, required)

    examples = {name: load_examples(path, args.limit) for name, path in SETS.items()}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "Qwen3.5-2B-Text",
                                              local_files_only=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prepared = {name: prepare_examples(items, tokenizer, 1024)
                for name, items in examples.items()}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("该评测需使用 GPU；当前未检测到 CUDA")

    print("加载基座模型与三个 LoRA adapter…", flush=True)
    full_model = Qwen3_5ForCausalLM.from_pretrained(
        ROOT / "models" / "Qwen3.5-2B-Text", dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, local_files_only=True,
        low_cpu_mem_usage=True)
    backbone = full_model.model
    del full_model
    backbone.config.use_cache = False
    first = checkpoint_names[0]
    backbone = PeftModel.from_pretrained(backbone, CHECKPOINTS[first] / "adapter",
                                        adapter_name=first, is_trainable=False)
    for name in checkpoint_names[1:]:
        backbone.load_adapter(CHECKPOINTS[name] / "adapter", adapter_name=name,
                              is_trainable=False)
    backbone.eval()
    head = ScoreHead(backbone.config.hidden_size).to(device)
    head.eval()
    free, total = torch.cuda.mem_get_info()
    print(f"模型已加载，剩余显存 {free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)

    all_metrics = {}
    for name in checkpoint_names:
        backbone.set_adapter(name)
        head_state = torch.load(CHECKPOINTS[name] / "score_head.pt",
                                map_location="cpu", weights_only=True)
        head.load_state_dict(head_state)
        print(f"开始评测 {name}", flush=True)
        all_metrics[name] = {}
        for set_name, items in prepared.items():
            prediction_path = args.output_dir / f"{name}_{set_name}_predictions.jsonl"
            all_metrics[name][set_name] = evaluate_one_set(
                backbone, head, items, args.candidate_batch_size,
                pad_token_id, device, prediction_path)
            overall = all_metrics[name][set_name]["overall"]
            print(f"  {set_name}: n={overall['count']}, Acc={overall['accuracy']:.4f}, "
                  f"NLL={overall['nll']:.4f}, Brier={overall['brier']:.4f}", flush=True)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "max_length": 1024,
        "candidate_batch_size": args.candidate_batch_size,
        "limit_per_set": args.limit,
        "checkpoint_names": checkpoint_names,
        "checkpoint_paths": {name: str(CHECKPOINTS[name]) for name in checkpoint_names},
        "test_set_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in SETS.items()},
        "prompt_builder": "jev/common.py:branch_prompts",
        "preprocessing": "jev/train_cluster.py:prepare_examples",
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("评测完成。", flush=True)


if __name__ == "__main__":
    main()
