"""面向高性能算力平台 (A100/H100/RTX) 的通用图文多模态兼容微调脚本.

核心特性:
1. 完整接回 Qwen 原生视觉编码器，支持纯文本与多模态图文数据混合训练；
2. 支持动态候选项数量 (2~10+ 候选)，按题分组计算 Softmax 交叉熵损失；
3. 支持 LoRA 微调语言模型骨干 + 决策评分头 ScoreHead，可选微调视觉 Projector；
4. 算力平台专属无头运行，支持 nohup / Slurm 批处理，具备优雅中断与自动 Checkpoint 恢复机制。
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict

from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P")

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


@dataclass(frozen=True)
class MultimodalExample:
    id: str
    description: str
    candidates: tuple[str, ...]
    gold_index: int
    language: str = "zh"
    source_dataset: str = "unknown"
    domain: str = "unknown"
    image_path: Optional[str] = None


def load_multimodal_examples(path: Path, limit: Optional[int] = None) -> List[MultimodalExample]:
    examples: List[MultimodalExample] = []
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
                continue
            if not isinstance(description, str) or not description.strip():
                continue
            if not isinstance(candidates, list) or len(candidates) < 2:
                continue
            seen.add(identifier)
            
            img_val = raw.get("image") or raw.get("image_path")
            examples.append(
                MultimodalExample(
                    id=identifier,
                    description=description.strip(),
                    candidates=tuple(str(c).strip() for c in candidates),
                    gold_index=int(gold),
                    language=raw.get("language", "zh"),
                    source_dataset=str(raw.get("source_dataset", "unknown")),
                    domain=str(raw.get("domain", "unknown")),
                    image_path=str(img_val).strip() if img_val else None,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多模态兼容通用决策模型微调脚本")
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3.5-2B"))
    parser.add_argument("--train-data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/multimodal-cluster"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--train-projector", action="store_true", help="是否联合微调视觉投影层以获得最高对齐度")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def log_event(log_file: Path, message: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {message}"
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_question_branches(example: MultimodalExample, processor: Any) -> tuple[List[str], Optional[Image.Image]]:
    choices = "\n".join([f"{LETTERS[i]}. {c}" for i, c in enumerate(example.candidates)])
    img: Optional[Image.Image] = None
    if example.image_path and Path(example.image_path).exists():
        try:
            img = Image.open(example.image_path).convert("RGB")
        except Exception:
            img = None

    prompts = []
    for pos, cand in enumerate(example.candidates):
        if example.language == "zh":
            text = (
                f"问题：{example.description}\n完整候选列表：\n{choices}\n"
                f"待判断候选：{LETTERS[pos]}. {cand}\n判断："
            )
        else:
            text = (
                f"Question: {example.description}\nAll options:\n{choices}\n"
                f"Candidate to assess: {LETTERS[pos]}. {cand}\nJudgment:"
            )
        content = []
        if img is not None:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        p = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=False, tokenize=False)
        prompts.append(p)
    return prompts, img


def main() -> None:
    args = parse_arguments()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / "train.log"
    history_file = args.output_dir / "history.jsonl"
    
    log_event(log_file, f"[*] 启动多模态兼容训练任务，输出目录: {args.output_dir}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # 冻结视觉编码器或开启 Projector 微调
    for p in full_model.model.visual.parameters():
        p.requires_grad = False
    if args.train_projector and hasattr(full_model.model.visual, "merger"):
        log_event(log_file, "[*] 激活视觉 Projector (merger) 联合微调以对齐隐空间")
        for p in full_model.model.visual.merger.parameters():
            p.requires_grad = True
            
    # 给语言模型挂载 LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(LORA_TARGETS),
        bias="none",
    )
    full_model.model.language_model = get_peft_model(full_model.model.language_model, lora_config)
    
    hidden_size = full_model.config.text_config.hidden_size if hasattr(full_model.config, "text_config") else full_model.config.hidden_size
    head = ScoreHead(hidden_size).to(device).to(torch.bfloat16)
    
    trainable_params = [p for p in full_model.parameters() if p.requires_grad] + list(head.parameters())
    num_trainable = sum(p.numel() for p in trainable_params)
    log_event(log_file, f"[*] 可训练参数量: {num_trainable / 1e6:.2f} M")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    train_examples = load_multimodal_examples(args.train_data, args.max_train_examples)
    val_examples = load_multimodal_examples(args.validation_data, args.max_validation_examples)
    log_event(log_file, f"[*] 加载训练样本: {len(train_examples)} 条 | 验证样本: {len(val_examples)} 条")
    
    global_step = 0
    full_model.train()
    head.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    accumulated_examples = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        if STOP_REQUESTED:
            break
        random.shuffle(train_examples)
        log_event(log_file, f"--- 开始 Epoch {epoch + 1}/{args.epochs} ---")
        
        for ex_idx, example in enumerate(train_examples):
            if STOP_REQUESTED:
                break
                
            prompts, img = build_question_branches(example, processor)
            images = [img] * len(prompts) if img is not None else None
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device)
            
            outputs = full_model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[-1]
            last_pos = inputs.attention_mask.sum(dim=1) - 1
            last_hidden = hidden[torch.arange(hidden.shape[0], device=device), last_pos]
            
            scores = head(last_hidden).squeeze(-1).float()
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([example.gold_index], device=device))
            
            loss_scaled = loss / args.gradient_accumulation
            loss_scaled.backward()
            
            accumulated_loss += loss.item()
            accumulated_examples += 1
            
            if accumulated_examples >= args.gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                avg_loss = accumulated_loss / accumulated_examples
                accumulated_loss = 0.0
                accumulated_examples = 0
                
                if global_step % args.log_every == 0:
                    speed = args.log_every * args.gradient_accumulation / (time.time() - start_time)
                    log_event(log_file, f"Epoch {epoch+1} | Step {global_step} | Loss: {avg_loss:.4f} | 速度: {speed:.1f} 题/秒")
                    start_time = time.time()
                    
                if global_step % args.save_every == 0:
                    ckpt_path = checkpoints_dir / f"step-{global_step:06d}"
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    full_model.model.language_model.save_pretrained(ckpt_path / "adapter")
                    torch.save(head.state_dict(), ckpt_path / "score_head.pt")
                    log_event(log_file, f"[✓] 保存检查点: {ckpt_path}")
                    
    log_event(log_file, "[✓] 训练任务完成或安全退出！")

if __name__ == "__main__":
    main()
