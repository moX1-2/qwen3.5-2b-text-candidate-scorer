---
language:
- zh
- en
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
pipeline_tag: text-classification
tags:
- qwen3_5
- candidate-scoring
- multiple-choice
- reranking
---

# Qwen3.5-2B Text Candidate Scorer：可变候选评分模型

本模型是基于 [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 的独立双语候选评分实验。它接收问题描述和 2 至 5 个候选项，为每项给出分数，并在当前候选集合内计算 Softmax 概率。项目受 TypeSafe AI 的 Jev 启发，**与 TypeSafe AI 无关联，也不提供其官方模型或服务**。

This is an independent Chinese and English candidate scoring experiment built on the Qwen3.5-2B text backbone. It ranks supplied options and does not generate free text. It is not affiliated with TypeSafe AI.

## 架构

![模型架构](docs/figures/qwen3.5-2b-text-candidate-scorer-architecture.png)

每个候选分支都包含问题、完整候选列表和当前待判断项。Qwen3.5-2B 文本骨干加载 LoRA 后，取分支最后一个有效 token 的隐藏状态，经含偏置的线性评分头输出标量。所有分支共用参数，但当前推理逐支计算，不复用公共前缀缓存。文本骨干从上游模型提取，排除了视觉与 MTP 权重。

本次发布包含裁剪后的文本骨干权重、8 万题阶段第 5,000 步的 LoRA、评分头、分词器和推理脚本。LoRA 参数为 r=8、alpha=16、dropout=0.05；每个候选分支的输入上限为 1,024 token。

## 快速试玩

需要 Linux、支持 BF16 的 NVIDIA CUDA GPU 和兼容的 Python 环境。项目在 RTX 4060 Laptop GPU（8 GiB）上完成过推理。

~~~bash
hf download mx-2026/qwen3.5-2b-text-candidate-scorer \
  --local-dir qwen3.5-2b-text-candidate-scorer
cd qwen3.5-2b-text-candidate-scorer
bash jev/setup_env.sh
jev/.venv/bin/python jev/play.py
~~~

交互时依次输入问题和候选项。输入完 2 至 5 个候选项后，在下一项直接回车，再输入要选择的项数。直接回车默认选择 1 项。脚本显示全部选项的相对概率、选中的前 N 项和本题耗时。单次命令示例：

~~~bash
jev/.venv/bin/python jev/play.py \
  --question "18 加 27 等于多少？" \
  --candidate 46 --candidate 45 --candidate 44 \
  --top-k 2
~~~

前 N 项选择只按单选模型的分数排序。模型未接受多正确项训练；组内概率不是校准后的现实正确率。本题耗时包含分词和推理，不包含人工输入与首次模型加载。

## 文件

~~~text
models/Qwen3.5-2B-Text/                         文本骨干、配置与分词器
result/unpacked/epoch-00-step-005000/adapter/     LoRA
result/unpacked/epoch-00-step-005000/score_head.pt 评分头
jev/play.py                                      交互与单次推理
jev/common.py、jev/train_cluster.py              提示构造与评分头定义
docs/figures/                                    架构图 PNG 与 SVG
~~~

推理脚本显式加载文本骨干、LoRA 和评分头。模型不支持直接调用 generate 或通用聊天接口。本仓库不包含原始题库、转换后的训练题、优化器状态或逐题留出题文本。

## 训练与评测

第一阶段使用 20,000 道双语题训练两轮；第二阶段保留这些题并扩充至 80,000 道。本次选择 8 万题第一轮末尾的第 5,000 步 checkpoint。两个阶段的细节见 [GitHub 全流程记录](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/docs/%E5%85%A8%E6%B5%81%E7%A8%8B%E5%A4%8D%E7%8E%B0.md)。

| 评测集 | 结果 | 说明 |
|---|---:|---|
| 同来源新留出题 | 169 / 200，准确率 84.5%，NLL 0.4418 | BoolQ、CommonsenseQA、ARC、C3、OCNLI 各 40 题 |
| 自拟单选题 | 20 / 20 | 小规模、较简单的功能测试 |

留出题来自既有训练来源的其他样本。这些结果不能代表跨来源泛化能力，也不是官方基准测试分数。模型可能随候选措辞、顺序和数量改变判断；它没有经过拒答、多正确项或长文本系统评测。详细分组与限制见 [评测报告](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/result/evaluation-2026-09-25/REPORT.md)。

## 来源与许可

文本骨干来自 [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，上游模型标注 Apache-2.0。公开代码和权重按本仓库所列许可发布。训练来源还包括 BoolQ、AI2 ARC、CommonsenseQA、MMLU、OCNLI、C3、C-Eval 和 CMMLU；各来源的数据使用条件需要分别遵守。本仓库不再分发这些训练题。

其中 [C-Eval 官方数据集](https://huggingface.co/datasets/ceval/ceval-exam)标注 CC BY-NC-SA 4.0。CMMLU 的使用条件及其他来源详见 [数据来源说明](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/docs/%E6%A8%A1%E5%9E%8B%E5%8D%A1.md)。本仓库的许可声明不改变第三方数据的条款；计划商业使用时，应由使用者逐项核对来源许可。

## 项目

- [GitHub 仓库](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer)
- [模型卡与发布说明](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/docs/%E6%A8%A1%E5%9E%8B%E5%8D%A1.md)
