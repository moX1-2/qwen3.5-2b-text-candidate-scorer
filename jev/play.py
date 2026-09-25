"""交互试玩 8 万题第 5,000 步的候选评分模型。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from common import Example, LETTERS, branch_prompts
from train_cluster import ScoreHead


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "Qwen3.5-2B-Text"
CHECKPOINT = ROOT / "result" / "unpacked" / "epoch-00-step-005000"
MAX_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="单次提问；省略时进入交互模式")
    parser.add_argument("--candidate", action="append", default=[], help="候选项，重复传入 2～5 次")
    parser.add_argument("--top-k", type=int, default=1, help="选择概率最高的前 N 项，默认 1")
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 5:
        parser.error("--top-k 必须为 1～5")
    if args.question is None and args.candidate:
        parser.error("使用 --candidate 时还需传入 --question")
    if args.question is not None:
        validate(args.question, args.candidate, parser.error)
        if args.top_k > len(args.candidate):
            parser.error("--top-k 不能大于候选项数量")
    return args


def validate(question: str, candidates: list[str], fail) -> None:
    if not question.strip():
        fail("问题不能为空")
    if not 2 <= len(candidates) <= 5:
        fail("请输入 2～5 个候选项")
    if any(not candidate.strip() for candidate in candidates):
        fail("候选项不能为空")


def load_model():
    for path in (MODEL / "model.safetensors", CHECKPOINT / "adapter" / "adapter_model.safetensors", CHECKPOINT / "score_head.pt"):
        if not path.is_file():
            raise FileNotFoundError(f"缺少模型文件：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU；此脚本需要 NVIDIA 显卡和 CUDA 环境")

    print("正在加载 80k_step5000 模型…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    full_model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        local_files_only=True, low_cpu_mem_usage=True,
    )
    backbone = full_model.model
    del full_model
    backbone.config.use_cache = False
    backbone = PeftModel.from_pretrained(backbone, CHECKPOINT / "adapter", is_trainable=False)
    backbone.eval()
    head = ScoreHead(backbone.config.hidden_size).to("cuda:0")
    head.load_state_dict(torch.load(CHECKPOINT / "score_head.pt", map_location="cpu", weights_only=True))
    head.eval()
    print("加载完成。\n", flush=True)
    return tokenizer, backbone, head


@torch.inference_mode()
def predict(question: str, candidates: list[str], language: str, tokenizer, backbone, head) -> list[float]:
    example = Example("interactive", question.strip(), tuple(item.strip() for item in candidates),
                      0, language, "interactive", "interactive")
    prompts = branch_prompts(example)
    # 直接使用单条编码接口，避开 Transformers 包装层的 encode_batch 路径。
    try:
        input_ids = [tokenizer.backend_tokenizer.encode(prompt, add_special_tokens=True).ids
                     for prompt in prompts]
    except TypeError:
        fresh_tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        input_ids = [fresh_tokenizer.backend_tokenizer.encode(prompt, add_special_tokens=True).ids
                     for prompt in prompts]
    lengths = [len(ids) for ids in input_ids]
    if max(lengths) > MAX_LENGTH:
        raise ValueError(f"输入为 {max(lengths)} token，超过训练时的 {MAX_LENGTH} token；请缩短题目或选项")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    scores = []
    for offset in range(0, len(candidates), 2):
        ids = [torch.tensor(item, dtype=torch.long) for item in input_ids[offset:offset + 2]]
        padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id).to("cuda:0")
        attention_mask = (padded != pad_id).long()
        hidden = backbone(input_ids=padded, attention_mask=attention_mask, use_cache=False).last_hidden_state
        last_positions = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[torch.arange(hidden.shape[0], device="cuda:0"), last_positions]
        scores.extend(head(last_hidden.float()).squeeze(-1).float().cpu().tolist())
    return torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).tolist()


def show_result(candidates: list[str], probabilities: list[float], top_k: int, elapsed: float) -> None:
    selected = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))[:top_k]
    for index, (candidate, probability) in enumerate(zip(candidates, probabilities)):
        print(f"{LETTERS[index]}. {candidate}  {probability:.1%}")
    choices = "、".join(f"{LETTERS[index]}. {candidates[index]}" for index in selected)
    print(f"模型选择（前 {top_k} 项）：{choices}")
    print(f"本题耗时：{elapsed:.2f} 秒\n")


def read_question(default_top_k: int) -> tuple[str, list[str], int] | None:
    question = input("问题（直接回车退出）：").strip()
    if not question:
        return None
    candidates = []
    for index in range(5):
        suffix = "，回车结束" if index >= 2 else ""
        candidate = input(f"候选 {LETTERS[index]}{suffix}：").strip()
        if not candidate:
            if len(candidates) >= 2:
                break
            print("至少需要两个非空候选项，请重新输入这道题。\n")
            return read_question(default_top_k)
        candidates.append(candidate)
    default = min(default_top_k, len(candidates))
    while True:
        raw = input(f"选择几项（1～{len(candidates)}，回车默认 {default}）：").strip()
        if not raw:
            return question, candidates, default
        if raw.isdecimal() and 1 <= int(raw) <= len(candidates):
            return question, candidates, int(raw)
        print(f"请输入 1～{len(candidates)} 之间的整数。")


def main() -> None:
    args = parse_args()
    tokenizer, backbone, head = load_model()
    if args.question is not None:
        started = time.perf_counter()
        probabilities = predict(args.question, args.candidate, args.language, tokenizer, backbone, head)
        show_result(args.candidate, probabilities, args.top_k, time.perf_counter() - started)
        return
    print("输入问题和 2～5 个候选项，再指定选择几项；按 Ctrl+C 也可退出。\n")
    try:
        while item := read_question(args.top_k):
            question, candidates, top_k = item
            try:
                started = time.perf_counter()
                probabilities = predict(question, candidates, args.language, tokenizer, backbone, head)
            except (TypeError, ValueError) as error:
                print(f"无法评分：{error}\n")
                continue
            show_result(candidates, probabilities, top_k, time.perf_counter() - started)
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")


if __name__ == "__main__":
    main()
