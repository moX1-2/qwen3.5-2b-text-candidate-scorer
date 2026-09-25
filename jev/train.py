from __future__ import annotations
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
"""Fine-tune a shared candidate score head, with optional text-only LoRA."""


import argparse
import gc
import json
import math
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from dashboard import TrainDashboard
import sys
from common import (
    DEFAULT_MODEL,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    branch_prompts,
    encode_texts,
    load_examples,
    stratified_metrics,
)


LORA_TARGETS = (
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
STOP_REQUESTED = False


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; will save after the current optimizer step", flush=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-data", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("head", "lora"), default="lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluate validation every N steps")
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--resume", help="Checkpoint directory, or auto for output-dir/latest.json")
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    for name in (
        "epochs",
        "max_length",
        "gradient_accumulation",
        "lora_r",
        "lora_alpha",
        "save_every",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_train_examples", "max_validation_examples", "max_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.lora_dropout < 1 or not 0 <= args.warmup_ratio < 1:
        parser.error("Dropout and warmup ratio must be in [0, 1)")
    return args


def resume_directory(args: argparse.Namespace) -> Path | None:
    if args.resume is None:
        return None
    if args.resume == "auto":
        pointer = args.output_dir / "latest.json"
        if not pointer.is_file():
            return None  # 首次运行无存档时自动从头开始，无缝接力
        return args.output_dir / json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
    return Path(args.resume).resolve()


def prepare(examples: list, tokenizer: Any, max_length: int) -> list[tuple[Any, Any]]:
    prepared = []
    for example in examples:
        batch = encode_texts(tokenizer, branch_prompts(example), max_length, example.id)
        prepared.append((example, batch))
    return prepared


def score_group(backbone: Any, head: nn.Module, batch: Any, train_backbone: bool) -> torch.Tensor:
    device = next(head.parameters()).device
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    if train_backbone:
        hidden = backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
    else:
        with torch.no_grad():
            hidden = backbone(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
    last_positions = attention_mask.sum(dim=1) - 1
    last_hidden = hidden[torch.arange(hidden.shape[0], device=device), last_positions]
    return head(last_hidden.float()).squeeze(-1)


def evaluate(backbone: Any, head: nn.Module, prepared: list, train_backbone: bool) -> dict:
    backbone.eval()
    head.eval()
    records = []
    with torch.inference_mode():
        for example, batch in prepared:
            scores = score_group(backbone, head, batch, train_backbone)
            probabilities = torch.softmax(scores.float(), dim=0).cpu().tolist()
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
            records.append(
                {
                    "id": example.id,
                    "language": example.language,
                    "source_dataset": example.source_dataset,
                    "domain": example.domain,
                    "candidate_count": len(example.candidates),
                    "gold_position": example.gold_index,
                    "predicted_position": predicted,
                    "correct": predicted == example.gold_index,
                    "probabilities": probabilities,
                }
            )
    torch.cuda.empty_cache()
    gc.collect()
    return {"metrics": stratified_metrics(records), "predictions": records}


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_checkpoint(
    args: argparse.Namespace,
    backbone: Any,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    next_epoch: int,
    next_position: int,
    global_step: int,
    best_nll: float | None,
) -> Path:
    directory = args.output_dir / "checkpoints" / f"epoch-{next_epoch:02d}-step-{global_step:06d}"
    if directory.exists():
        raise FileExistsError(f"Checkpoint already exists: {directory}")
    directory.mkdir(parents=True)
    if args.method == "lora":
        backbone.save_pretrained(directory / "adapter", safe_serialization=True)
    torch.save(head.state_dict(), directory / "score_head.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        directory / "optimizer.pt",
    )
    state = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "method": args.method,
        "next_epoch": next_epoch,
        "next_position": next_position,
        "global_step": global_step,
        "best_nll": best_nll,
        "seed": args.seed,
    }
    (directory / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pointer = args.output_dir / "latest.json"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"checkpoint": str(directory.relative_to(args.output_dir))}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(pointer)
    print(f"Saved {directory}", flush=True)
    return directory


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an allocated CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    train_examples = load_examples(args.train_data, args.max_train_examples)
    validation_examples = load_examples(args.validation_data, args.max_validation_examples)
    overlap = {example.id for example in train_examples} & {
        example.id for example in validation_examples
    }
    if overlap:
        raise ValueError(f"Train/validation id overlap: {next(iter(overlap))}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    training = prepare(train_examples, tokenizer, args.max_length)
    validation = prepare(validation_examples, tokenizer, args.max_length)
    print(
        f"Prepared {len(training)} training groups and {len(validation)} validation groups",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = resume_directory(args)
    if checkpoint_dir is None and (args.output_dir / "latest.json").exists():
        raise FileExistsError("Output has a checkpoint; use --resume auto or a new output directory")
    saved_state = None
    if checkpoint_dir is not None:
        saved_state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
        for key, expected in (
            ("model", str(args.model.resolve())),
            ("train_data", str(args.train_data.resolve())),
            ("validation_data", str(args.validation_data.resolve())),
            ("method", args.method),
            ("seed", args.seed),
        ):
            if saved_state[key] != expected:
                raise ValueError(f"Checkpoint {key} differs from this run")

    full_model = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    backbone = full_model.model
    del full_model
    gc.collect()
    backbone.config.use_cache = False
    head = nn.Linear(backbone.config.hidden_size, 1, bias=False, dtype=torch.float32).cuda()

    if args.method == "head":
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.eval()
    else:
        present = {
            name.rsplit(".", 1)[-1]
            for name, module in backbone.named_modules()
            if isinstance(module, nn.Linear)
        }
        missing = set(LORA_TARGETS) - present
        if missing:
            raise RuntimeError(f"Expected text LoRA targets are missing: {sorted(missing)}")
        backbone.enable_input_require_grads()
        if checkpoint_dir is None:
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=list(LORA_TARGETS),
                    bias="none",
                ),
            )
        else:
            backbone = PeftModel.from_pretrained(
                backbone, checkpoint_dir / "adapter", is_trainable=True
            )
        if args.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        backbone.train()
    if checkpoint_dir is not None:
        head.load_state_dict(
            torch.load(checkpoint_dir / "score_head.pt", map_location="cpu", weights_only=True)
        )

    adapter_parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in adapter_parameters) + sum(
        parameter.numel() for parameter in head.parameters()
    )
    print(
        json.dumps(
            {
                "method": args.method,
                "trainable_parameters": trainable_count,
                "total_text_parameters": sum(parameter.numel() for parameter in backbone.parameters()),
                "gpu": torch.cuda.get_device_name(0),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    parameter_groups = []
    if adapter_parameters:
        parameter_groups.append({"params": adapter_parameters, "lr": args.adapter_lr})
    parameter_groups.append({"params": list(head.parameters()), "lr": args.head_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(training) / args.gradient_accumulation)
    scheduler = make_scheduler(optimizer, updates_per_epoch * args.epochs, args.warmup_ratio)

    start_epoch = 0
    start_position = 0
    global_step = 0
    best_nll = None
    if checkpoint_dir is not None and saved_state is not None:
        state = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start_epoch = int(saved_state["next_epoch"])
        start_position = int(saved_state["next_position"])
        global_step = int(saved_state["global_step"])
        best_nll = saved_state["best_nll"]

    torch.cuda.reset_peak_memory_stats()
    total_steps = args.max_steps if args.max_steps else (len(training) // args.gradient_accumulation) * args.epochs
    dashboard = None
    if args.ui:
        dashboard = TrainDashboard(
            total_examples=len(training) * args.epochs if not args.max_steps else args.max_steps * args.gradient_accumulation,
            total_steps=total_steps,
            method=args.method,
            model_name=args.model.name,
            gradient_accumulation=args.gradient_accumulation,
        )
        dashboard.start()
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(training)))
        random.Random(args.seed + epoch).shuffle(order)
        position = start_position if epoch == start_epoch else 0
        backbone.train(args.method == "lora")
        head.train()
        optimizer.zero_grad(set_to_none=True)
        while position < len(order):
            window = order[position : position + args.gradient_accumulation]
            window_loss = 0.0
            window_correct = 0
            for index in window:
                example, batch = training[index]
                scores = score_group(backbone, head, batch, args.method == "lora")
                gold = torch.tensor([example.gold_index], device=scores.device)
                loss = F.cross_entropy(scores.unsqueeze(0).float(), gold)
                (loss / len(window)).backward()
                window_loss += float(loss.detach())
                window_correct += int(scores.argmax().item() == example.gold_index)
            torch.nn.utils.clip_grad_norm_(
                adapter_parameters + list(head.parameters()), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            position += len(window)
            current_peak_gpu = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            # 自动持久化落盘每一步的 loss 历史
            h_record = {
                "step": global_step,
                "loss": round(window_loss / len(window), 5),
                "accuracy": round(window_correct / len(window), 4),
                "lr": round(scheduler.get_last_lr()[0] if scheduler else 0.0, 7),
                "peak_gpu_gib": current_peak_gpu,
            }
            with open(args.output_dir / "history.jsonl", "a", encoding="utf-8") as hf:
                hf.write(json.dumps(h_record, ensure_ascii=False) + "\n")

            if dashboard:
                dashboard.update_step(
                    step=global_step,
                    examples_seen=position,
                    loss=window_loss / len(window),
                    acc=window_correct / len(window),
                    lr=scheduler.get_last_lr()[0] if scheduler else 0.0,
                    peak_gpu=round(torch.cuda.memory_allocated() / 2**30, 2),
                )
            if global_step % 20 == 0:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            if global_step % args.log_every == 0 or global_step == 1:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "groups_seen_this_epoch": position,
                            "loss": window_loss / len(window),
                            "accuracy": window_correct / len(window),
                            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if args.eval_every and global_step % args.eval_every == 0:
                eval_subset = validation[:args.max_validation_examples] if args.max_validation_examples else validation
                val_result = evaluate(backbone, head, eval_subset, args.method == "lora")
                val_overall = val_result["metrics"]["overall"]
                if dashboard:
                    dashboard.update_eval(global_step, val_overall)
                else:
                    print(json.dumps({"step": global_step, "validation": val_overall}, ensure_ascii=False), flush=True)
            if global_step % args.save_every == 0:
                save_checkpoint(
                    args, backbone, head, optimizer, scheduler, epoch, position, global_step, best_nll
                )
            if STOP_REQUESTED or (args.max_steps is not None and global_step >= args.max_steps):
                if global_step % args.save_every:
                    save_checkpoint(
                        args, backbone, head, optimizer, scheduler, epoch, position, global_step, best_nll
                    )
                return

        result = evaluate(backbone, head, validation, args.method == "lora")
        overall = result["metrics"]["overall"]
        (args.output_dir / f"validation-epoch-{epoch + 1}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"epoch": epoch + 1, "validation": overall}, ensure_ascii=False), flush=True)
        improved = best_nll is None or overall["nll"] < best_nll
        if improved:
            best_nll = overall["nll"]
        checkpoint = save_checkpoint(
            args, backbone, head, optimizer, scheduler, epoch + 1, 0, global_step, best_nll
        )
        if improved:
            (args.output_dir / "best.json").write_text(
                json.dumps({"checkpoint": str(checkpoint.relative_to(args.output_dir))}, indent=2) + "\n",
                encoding="utf-8",
            )
        start_position = 0

    if dashboard:
        dashboard.stop()
    print(
        json.dumps(
            {
                "completed_epochs": args.epochs,
                "optimizer_steps": global_step,
                "best_validation_nll": best_nll,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
