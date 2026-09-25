"""面向高性能算力平台 (A100/H100) 的专享微调脚本 (后台无阻塞运行版本).

特性:
1. 彻底移除前台 ANSI 终端刷新看板，支持 nohup / sbatch 无头静默运行与可靠日志记录。
2. 算力卡级真·批处理 (Batched Forward & Backward): 16 题 (~64 个候选序列) 打包成单一批次执行。
3. 动态 Padding 裁剪 (Dynamic Batch Padding): 仅填充至当前批次内的最大长度，避免无效计算。
4. 深度释放 A100 硬件性能: 强制激活 TF32 / BF16 硬件张量核心加速，默认关闭梯度检查点 (显存充裕)。
5. 独立结构: 兼容原 checkpoint 格式，与本地原始 train.py 隔离解耦。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# A100 / H100 算力内核全局硬件加速
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from common import (
    DEFAULT_MODEL,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    Example,
    branch_prompts,
    load_examples,
    stratified_metrics,
)

LORA_TARGETS = (
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
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


def handle_signal(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[INFO] 收到终止信号，正在准备安全保存 Checkpoint...", flush=True)


class ScoreHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A100 专属无头后台训练脚本")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-data", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cluster-a100"))
    parser.add_argument("--method", choices=("probe", "lora"), default="lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation", type=int, default=16, help="每步累积题数 (默认 16 题)")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="单次前向打包题数 (A100 上推荐 16, 一步 1s)")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-batch-size", type=int, default=16, help="验证集评测打包题数 (推荐 16~32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=100, help="评测间隔步数")
    parser.add_argument("--save-every", type=int, default=50, help="检查点保存间隔步数")
    parser.add_argument("--log-every", type=int, default=10, help="日志记录间隔步数")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                        help="梯度检查点 (A100 显存充裕默认关闭，提速 40%%)")
    parser.add_argument("--resume", default="auto", help="'auto'、检查点目录或 'none'")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="继承检查点权重但重置优化器与学习率调度器（用于开启新阶段微调或更换数据集）")
    parser.add_argument("--keep-checkpoints", type=int, default=0, help="最多保留最近几个历史检查点 (0 表示完全保留所有检查点，绝不自动删除)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_examples(examples: list[Example], tokenizer: Any, max_length: int) -> list[tuple[Example, list[torch.Tensor]]]:
    prepared = []
    for example in examples:
        texts = branch_prompts(example)
        encoded = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        item_tensors = [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]
        prepared.append((example, item_tensors))
    return prepared


def collate_micro_batch(
    items: list[tuple[Example, list[torch.Tensor]]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    candidate_counts = []
    golds = []
    all_input_ids = []
    for example, item_tensors in items:
        candidate_counts.append(len(item_tensors))
        golds.append(example.gold_index)
        all_input_ids.extend(item_tensors)

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        all_input_ids, batch_first=True, padding_value=pad_token_id
    )
    padded_attention_mask = (padded_input_ids != pad_token_id).long()
    return padded_input_ids, padded_attention_mask, candidate_counts, golds


def score_batched(
    backbone: Any,
    head: nn.Module,
    padded_input_ids: torch.Tensor,
    padded_attention_mask: torch.Tensor,
    candidate_counts: list[int],
    train_backbone: bool,
) -> list[torch.Tensor]:
    device = next(head.parameters()).device
    input_ids = padded_input_ids.to(device, non_blocking=True)
    attention_mask = padded_attention_mask.to(device, non_blocking=True)

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
    all_scores = head(last_hidden.float()).squeeze(-1)
    return list(torch.split(all_scores, candidate_counts))


def evaluate_cluster(
    backbone: Any,
    head: nn.Module,
    prepared: list[tuple[Example, list[torch.Tensor]]],
    batch_size: int,
    pad_token_id: int,
) -> dict[str, Any]:
    backbone.eval()
    head.eval()
    records = []
    with torch.inference_mode():
        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i : i + batch_size]
            input_ids, attention_mask, candidate_counts, golds = collate_micro_batch(
                chunk, pad_token_id=pad_token_id
            )
            scores_list = score_batched(
                backbone, head, input_ids, attention_mask, candidate_counts, train_backbone=False
            )
            for (example, _), scores, gold in zip(chunk, scores_list, golds):
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
                        "correct": predicted == gold,
                        "probabilities": probabilities,
                    }
                )
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
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    if args.method == "lora":
        backbone.save_pretrained(directory / "adapter", safe_serialization=True)
    torch.save(head.state_dict(), directory / "score_head.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        directory / "optimizer.pt",
    )
    metadata = {
        "method": args.method,
        "next_epoch": next_epoch,
        "next_position": next_position,
        "global_step": global_step,
        "best_nll": best_nll,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    (directory / "state.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    latest_tmp = args.output_dir / "latest.json.tmp"
    latest_tmp.write_text(
        json.dumps({"checkpoint": str(directory.relative_to(args.output_dir))}, indent=2) + "\n",
        encoding="utf-8",
    )
    latest_tmp.replace(args.output_dir / "latest.json")

    # 长线运行保护: 自动轮转清理旧检查点，保留最近 N 个，防止撑爆学校磁盘配额
    if getattr(args, "keep_checkpoints", 0) and args.keep_checkpoints > 0:
        ckpts_root = args.output_dir / "checkpoints"
        if ckpts_root.exists():
            all_ckpts = sorted([p for p in ckpts_root.iterdir() if p.is_dir() and p.name.startswith("epoch-")])
            if len(all_ckpts) > args.keep_checkpoints:
                to_remove = all_ckpts[:-args.keep_checkpoints]
                for old_dir in to_remove:
                    try:
                        shutil.rmtree(old_dir)
                    except Exception:
                        pass
    return directory


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume == "none":
        return None
    if args.resume == "auto":
        latest_file = args.output_dir / "latest.json"
        if not latest_file.exists():
            return None
        payload = json.loads(latest_file.read_text(encoding="utf-8"))
        candidate = args.output_dir / payload["checkpoint"]
        return candidate if candidate.exists() else None
    direct = Path(args.resume)
    if direct.exists():
        return direct
    relative = args.output_dir / args.resume
    if relative.exists():
        return relative
    raise FileNotFoundError(f"未找到断点续传检查点: {args.resume}")


def log_event(log_file: Path, text_msg: str, metric_data: dict[str, Any] | None = None) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_line = f"[{now_str}] {text_msg}"
    print(formatted_line, flush=True)

    with log_file.open("a", encoding="utf-8") as f:
        f.write(formatted_line + "\n")

    if metric_data is not None:
        metric_file = log_file.parent / "metrics.jsonl"
        with metric_file.open("a", encoding="utf-8") as f:
            metric_entry = {"timestamp": now_str, **metric_data}
            f.write(json.dumps(metric_entry, ensure_ascii=False) + "\n")


def main() -> None:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / "train.log"

    # 保存启动 PID
    (args.output_dir / "train.pid").write_text(str(os.getpid()), encoding="utf-8")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_total = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        if torch.cuda.is_available()
        else 0.0
    )

    log_event(log_file, "=" * 70)
    log_event(log_file, f"A100 专享算力集群训练启动 | 目标架构: {args.method.upper()}")
    log_event(log_file, f"计算硬件: {device_name} (总显存: {vram_total} GiB) | PID: {os.getpid()}")
    log_event(
        log_file,
        f"超参配置: 累积={args.gradient_accumulation}题/步, MicroBatch={args.micro_batch_size}题/前向, 梯度检查点={'开' if args.gradient_checkpointing else '关'}",
    )
    log_event(log_file, "=" * 70)

    # 1. 加载 Tokenizer 与 模型
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    full_model = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    backbone = full_model.model
    del full_model
    gc.collect()
    backbone.config.use_cache = False
    head = ScoreHead(backbone.config.hidden_size).to("cuda:0" if torch.cuda.is_available() else "cpu")

    resume_checkpoint = resolve_resume_checkpoint(args)
    saved_state = None
    if resume_checkpoint:
        log_event(log_file, f"正在从检查点恢复: {resume_checkpoint}")
        state_file = resume_checkpoint / "state.json"
        if state_file.exists():
            saved_state = json.loads(state_file.read_text(encoding="utf-8"))
        head_path = resume_checkpoint / "score_head.pt"
        if head_path.exists():
            head.load_state_dict(torch.load(head_path, map_location="cuda:0", weights_only=True))

    if args.method == "lora":
        backbone.enable_input_require_grads()
        if resume_checkpoint and (resume_checkpoint / "adapter").exists():
            backbone = PeftModel.from_pretrained(
                backbone,
                resume_checkpoint / "adapter",
                is_trainable=True,
            )
        else:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=list(LORA_TARGETS),
                bias="none",
            )
            backbone = get_peft_model(backbone, lora_config)

        if args.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        backbone.train()
        adapter_params = [p for p in backbone.parameters() if p.requires_grad]
    else:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        adapter_params = []

    # 2. 准备数据集
    log_event(log_file, "正在加载并预处理训练集与验证集...")
    raw_train = load_examples(args.train_data, args.max_train_examples)
    raw_val = load_examples(args.validation_data, args.max_validation_examples)
    training = prepare_examples(raw_train, tokenizer, args.max_length)
    validation = prepare_examples(raw_val, tokenizer, args.max_length)
    del raw_train, raw_val
    gc.collect()
    log_event(log_file, f"预处理完成: 训练集 {len(training)} 题, 验证集 {len(validation)} 题 (已释放原始内存)")

    steps_per_epoch = math.ceil(len(training) / args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    trainable_params = adapter_params + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)

    start_epoch = 0
    start_position = 0
    global_step = 0
    best_nll = None

    is_internal_resume = False
    if resume_checkpoint:
        try:
            is_internal_resume = args.output_dir.resolve() in resume_checkpoint.resolve().parents
        except Exception:
            pass

    should_reset_optimizer = args.reset_optimizer and not is_internal_resume

    if resume_checkpoint and (resume_checkpoint / "optimizer.pt").exists() and not should_reset_optimizer:
        saved_opt = torch.load(resume_checkpoint / "optimizer.pt", map_location="cpu")
        optimizer.load_state_dict(saved_opt["optimizer"])
        scheduler.load_state_dict(saved_opt["scheduler"])
        torch.set_rng_state(saved_opt["cpu_rng"])
        if torch.cuda.is_available() and "cuda_rng" in saved_opt and saved_opt["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved_opt["cuda_rng"])
        if saved_state:
            start_epoch = saved_state.get("next_epoch", 0)
            start_position = saved_state.get("next_position", 0)
            global_step = saved_state.get("global_step", 0)
            best_nll = saved_state.get("best_nll")
            log_event(log_file, f"断点恢复成功: 步数={global_step}/{total_steps}, Epoch={start_epoch + 1}, 已读样本={start_position}")
    elif resume_checkpoint and should_reset_optimizer:
        log_event(log_file, f"权重热启成功 (外部模型继承): 继承 LoRA 与 Score Head 权重，全新启动优化器与调度器，总目标步数={total_steps}")
    elif resume_checkpoint and is_internal_resume and args.reset_optimizer:
        log_event(log_file, f"检测到当前任务内部断点，自动忽略 --reset-optimizer，继续无缝断点续训！")

    started_at = time.monotonic()
    log_event(log_file, f"正式开始训练: 总步数={total_steps}, 每步处理 {args.gradient_accumulation} 题")

    for epoch in range(start_epoch, args.epochs):
        backbone.train()
        head.train()

        # 确定题目次序
        rng = random.Random(args.seed + epoch)
        order = list(range(len(training)))
        rng.shuffle(order)
        position = start_position

        while position < len(order):
            step_start_time = time.monotonic()
            window_indices = order[position : position + args.gradient_accumulation]
            window_len = len(window_indices)
            window_loss = 0.0
            window_correct = 0

            # 按照 micro_batch_size 对这 16 题进行真·批处理 (A100 上 micro_batch_size=16 即单次直接完成)
            mb_size = max(1, args.micro_batch_size)
            for chunk_offset in range(0, window_len, mb_size):
                chunk_idx = window_indices[chunk_offset : chunk_offset + mb_size]
                chunk_items = [training[i] for i in chunk_idx]

                padded_ids, padded_mask, candidate_counts, golds = collate_micro_batch(
                    chunk_items, pad_token_id=pad_token_id
                )

                scores_list = score_batched(
                    backbone,
                    head,
                    padded_ids,
                    padded_mask,
                    candidate_counts,
                    train_backbone=args.method == "lora",
                )

                device = next(head.parameters()).device
                chunk_loss = sum(
                    F.cross_entropy(s.unsqueeze(0).float(), torch.tensor([g], device=device))
                    for s, g in zip(scores_list, golds)
                )

                # 梯度按整个 window 进行归一化
                (chunk_loss / window_len).backward()
                window_loss += float(chunk_loss.detach())

                for s, g in zip(scores_list, golds):
                    window_correct += int(s.argmax().item() == g)

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_elapsed = time.monotonic() - step_start_time
            samples_per_sec = window_len / max(step_elapsed, 1e-5)
            global_step += 1
            position += window_len

            step_loss = window_loss / window_len
            step_acc = window_correct / window_len
            current_lr = scheduler.get_last_lr()[0]
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0

            # 结构化指标记录
            metric_record = {
                "step": global_step,
                "total_steps": total_steps,
                "epoch": epoch + 1,
                "progress_percent": round(global_step / total_steps * 100, 1),
                "loss": round(step_loss, 4),
                "accuracy": round(step_acc, 4),
                "step_seconds": round(step_elapsed, 3),
                "samples_per_sec": round(samples_per_sec, 1),
                "vram_gib": peak_vram,
                "lr": current_lr,
            }

            if global_step % args.log_every == 0 or global_step == 1:
                log_event(
                    log_file,
                    f"Step {global_step:>5d}/{total_steps} ({metric_record['progress_percent']:>4.1f}%) | "
                    f"Loss: {step_loss:.4f} | Acc: {step_acc*100:>5.1f}% | "
                    f"{step_elapsed:.2f}s/step ({samples_per_sec:>4.1f} 题/s) | 显存: {peak_vram:.1f}G",
                    metric_data=metric_record,
                )

            # 周期性验证集评估
            if args.eval_every and global_step % args.eval_every == 0:
                eval_start = time.monotonic()
                eval_subset = validation[:args.max_validation_examples] if args.max_validation_examples else validation
                val_result = evaluate_cluster(
                    backbone, head, eval_subset, batch_size=args.eval_batch_size, pad_token_id=pad_token_id
                )
                eval_dur = time.monotonic() - eval_start
                val_overall = val_result["metrics"]["overall"]
                log_event(
                    log_file,
                    f"[EVAL] Step {global_step} 评测完成 (耗时 {eval_dur:.1f}s): Acc={val_overall['accuracy']*100:.2f}%, NLL={val_overall['nll']:.4f}",
                    metric_data={"val_step": global_step, **val_overall},
                )
                backbone.train()
                head.train()

            # 周期性 Checkpoint 落盘
            if global_step % args.save_every == 0:
                ckpt_dir = save_checkpoint(
                    args, backbone, head, optimizer, scheduler, epoch, position, global_step, best_nll
                )
                log_event(log_file, f"[CKPT] 自动落盘检查点: {ckpt_dir.name}")

            if STOP_REQUESTED or (args.max_steps is not None and global_step >= args.max_steps):
                ckpt_dir = save_checkpoint(
                    args, backbone, head, optimizer, scheduler, epoch, position, global_step, best_nll
                )
                log_event(log_file, f"[STOP] 任务停止，已安全落盘检查点: {ckpt_dir.name}")
                return

        start_position = 0

    final_ckpt = save_checkpoint(
        args, backbone, head, optimizer, scheduler, args.epochs, len(training), global_step, best_nll
    )
    log_event(log_file, f"[FINAL] 训练全部圆满完成，最终检查点已安全落盘: {final_ckpt.name} (总耗时: {round((time.monotonic() - started_at)/60, 2)} 分钟)")


if __name__ == "__main__":
    main()
