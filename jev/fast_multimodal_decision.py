import copy
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer
from peft import PeftModel


class ScoreHead(nn.Module):
    """
    标量评分头：将骨干末位隐状态映射为决策分数
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


def clone_and_expand_hybrid_cache(cache: DynamicCache, num_branches: int) -> DynamicCache:
    """
    针对 Qwen3.5 独特混合架构（Gated DeltaNet + Conv + Full Attention）的缓存分支扩展器。
    将批大小为 1 的公共前缀缓存沿 Batch 维度克隆扩展为 num_branches 份。
    """
    expanded_cache = DynamicCache()
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionLayer):
            new_layer = LinearAttentionLayer(
                conv_kernel_size=layer.conv_kernel_size,
                dtype=layer.dtype,
                device=layer.device
            )
            # 复制并扩展卷积历史状态 [1, dim, kernel] -> [K, dim, kernel]
            new_layer.conv_states = {
                k: v.repeat(num_branches, 1, 1) for k, v in layer.conv_states.items()
            }
            # 复制并扩展 DeltaNet 循环状态 [1, heads, d1, d2] -> [K, heads, d1, d2]
            new_layer.recurrent_states = {
                k: v.repeat(num_branches, 1, 1, 1) for k, v in layer.recurrent_states.items()
            }
            expanded_cache.layers.append(new_layer)
        elif isinstance(layer, DynamicLayer):
            new_layer = DynamicLayer(
                dtype=layer.dtype,
                device=layer.device
            )
            # 复制并扩展自注意力 Key/Value [1, heads, seq, dim] -> [K, heads, seq, dim]
            new_layer.keys = layer.keys.repeat(num_branches, 1, 1, 1)
            new_layer.values = layer.values.repeat(num_branches, 1, 1, 1)
            expanded_cache.layers.append(new_layer)
        else:
            expanded_cache.layers.append(copy.deepcopy(layer))
    return expanded_cache


class FastMultimodalDecisionEngine:
    """
    低延迟多模态决策推理引擎 (Mugi Decision Multimodal Fast Engine)
    
    核心特性：
    1. 完整接回 Qwen 原生视觉编码器与投影层，复用基模图文对齐先验；
    2. 支持加载纯文本微调得到的 LoRA 与单标量 ScoreHead；
    3. 基于两阶段混合缓存（Prefix Prefill + Branch Forward with Shared Hybrid Cache），
       单次前向完成任意动态 K 个候选项的并行打分与归一化，消灭多模态长前缀的重复计算！
    """
    def __init__(
        self,
        base_model_path: str = "/home/moxi/graduate-project/new/models/Qwen3.5-2B",
        adapter_path: Optional[str] = "/home/moxi/graduate-project/new/result/unpacked/epoch-01-step-002500/adapter",
        score_head_path: Optional[str] = "/home/moxi/graduate-project/new/result/unpacked/epoch-01-step-002500/score_head.pt",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"[*] 正在初始化 Processor: {base_model_path}")
        self.processor = AutoProcessor.from_pretrained(base_model_path)
        
        print(f"[*] 正在加载 Qwen3.5 完整多模态基模（含视觉编码器）...")
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        self.model.eval()
        
        # 挂载 LoRA
        if adapter_path and Path(adapter_path).exists():
            print(f"[*] 正在挂载微调 LoRA 权重到语言模型骨干: {adapter_path}")
            peft_lm = PeftModel.from_pretrained(
                self.model.model.language_model,
                adapter_path
            )
            self.model.model.language_model = peft_lm
        
        # 挂载 ScoreHead
        hidden_size = self.model.config.text_config.hidden_size if hasattr(self.model.config, "text_config") else self.model.config.hidden_size
        self.score_head = ScoreHead(hidden_size).to(self.device).to(self.torch_dtype)
        if score_head_path and Path(score_head_path).exists():
            print(f"[*] 正在加载 ScoreHead 权重: {score_head_path}")
            state_dict = torch.load(score_head_path, map_location=self.device)
            self.score_head.load_state_dict(state_dict)
        self.score_head.eval()
        print("[✓] 多模态低延迟决策引擎初始化完成！\n")

    @torch.no_grad()
    def predict_fast(
        self,
        image: Optional[Image.Image],
        question: str,
        candidates: List[str],
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        极速决策分支预测：两阶段共享混合缓存
        """
        start_time = time.time()
        
        # 1. 构造公共前缀 Prompt
        enum_list = "\n".join([f"- {c}" for c in candidates])
        prompt_text = f"问题：{question}\n完整候选列表：\n{enum_list}\n"
        
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt_text})
        
        messages = [{"role": "user", "content": content}]
        prefix_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        )
        
        images = [image] if image is not None else None
        prefix_inputs = self.processor(
            text=[prefix_prompt],
            images=images,
            return_tensors="pt"
        ).to(self.device)
        
        # 阶段一：前缀预填充 (Prefix Prefill)
        prefix_out = self.model(**prefix_inputs, use_cache=True)
        prefix_cache = prefix_out.past_key_values
        prefix_len = prefix_inputs.input_ids.shape[1]
        
        # 阶段二：并行分支决策 (Parallel Branch Scoring)
        num_candidates = len(candidates)
        branch_cache = clone_and_expand_hybrid_cache(prefix_cache, num_candidates)
        
        # 构造分支后缀
        branch_texts = [f"待判断候选：{c}\n判断：" for c in candidates]
        branch_token_lists = [
            self.processor.tokenizer.encode(s, add_special_tokens=False)
            for s in branch_texts
        ]
        
        max_branch_len = max(len(t) for t in branch_token_lists)
        pad_id = self.processor.tokenizer.pad_token_id or 0
        
        padded_ids = []
        valid_lengths = []
        for t in branch_token_lists:
            valid_lengths.append(len(t))
            padded_ids.append(t + [pad_id] * (max_branch_len - len(t)))
            
        branch_input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        
        # 构建跨前缀与分支的完整注意力掩码
        total_len = prefix_len + max_branch_len
        attention_mask = torch.ones((num_candidates, total_len), device=self.device)
        for i, vlen in enumerate(valid_lengths):
            if vlen < max_branch_len:
                attention_mask[i, prefix_len + vlen:] = 0
                
        # 单次并行前向
        lm = self.model.model.language_model
        branch_out = lm(
            input_ids=branch_input_ids,
            attention_mask=attention_mask,
            past_key_values=branch_cache,
            use_cache=False,
            output_hidden_states=True
        )
        
        last_hidden_states = branch_out.hidden_states[-1]
        branch_vectors = []
        for i, vlen in enumerate(valid_lengths):
            branch_vectors.append(last_hidden_states[i, vlen - 1, :])
        branch_vectors = torch.stack(branch_vectors, dim=0)
        
        # 决策头打分与 Softmax 归一化
        scores = self.score_head(branch_vectors).squeeze(-1)
        probs = torch.softmax(scores / temperature, dim=-1)
        
        best_idx = torch.argmax(scores).item()
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "best_candidate": candidates[best_idx],
            "best_index": best_idx,
            "scores": scores.tolist(),
            "probabilities": probs.tolist(),
            "latency_ms": elapsed_ms,
            "prefix_tokens": prefix_len,
            "candidates_count": num_candidates,
            "mode": "fast_shared_hybrid_cache"
        }
