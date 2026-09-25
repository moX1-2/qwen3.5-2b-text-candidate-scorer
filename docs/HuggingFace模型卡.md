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

中文说明见下文；[English guide](#english-guide) 位于页面后半部分。

## 作者说明

这是作者首次从数据整理、模型结构适配到长程训练和本地推理完整运行的候选评分模型实验。第 5,000 步权重在算力平台完成训练，并在本机 RTX 4060 Laptop GPU 上完成固定题集推理。GitHub 保留训练与评测过程；本模型仓库提供下载后可运行的推理文件。

**环境适配提示：** 安装和推理已在 Linux、NVIDIA CUDA 与 RTX 4060 Laptop GPU 上验证。使用其他显卡或操作系统时，可以善用 AI 编码助手检查依赖、设备映射和内核兼容性，并用固定样例核对模型输出。当前没有其他平台的完整兼容性或性能数据。

**Environment note:** The published path was verified on Linux with NVIDIA CUDA and an RTX 4060 Laptop GPU. For another GPU or operating system, an AI coding assistant can help adapt dependencies, device placement, and kernels. Verify the resulting predictions against the sample input; compatibility and speed on other platforms have not been measured.

## 模型与输入

文本骨干从 Qwen3.5-2B 提取，包含 1,881,825,088 个参数、24 层解码器、2,048 维隐藏状态；其中 18 层使用线性注意力，6 层使用全注意力。训练更新 LoRA 与含偏置的线性评分头。输入沿用 Qwen 分词器，并按普通文本构造每个候选分支：

~~~text
问题：{问题描述}
完整候选列表：
A. {候选 A}
B. {候选 B}
...
待判断候选：A. {候选 A}
判断：
~~~

英文题使用对应的英文提示。每个分支都重复完整候选列表，只改变“待判断候选”。脚本按 2 个候选分支一批执行前向计算；每个分支最多 1,024 token，超出时会要求缩短输入。

## 架构图

![模型架构](docs/figures/qwen3.5-2b-text-candidate-scorer-architecture.png)

每个候选分支都包含问题、完整候选列表和当前待判断项。Qwen3.5-2B 文本骨干加载 LoRA 后，取分支最后一个有效 token 的隐藏状态，经含偏置的线性评分头输出标量。所有分支共用参数，但当前推理逐支计算，不复用公共前缀缓存。文本骨干从上游模型提取，排除了视觉与 MTP 权重。

本次发布包含裁剪后的文本骨干权重、8 万题阶段第 5,000 步的 LoRA、评分头、分词器和推理脚本。LoRA 参数为 r=8、alpha=16、dropout=0.05。

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

## 训练与权重选择

第一阶段使用 20,000 道双语题训练两轮，共 2,500 步。第二阶段继承 LoRA 与评分头，扩充到 80,000 道题，并重置优化器；第一轮在第 5,000 步结束，之后训练到第 6,500 步暂停。本模型仓库发布第 5,000 步权重。两个阶段的配置与恢复过程见 [GitHub 全流程记录](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/docs/%E5%85%A8%E6%B5%81%E7%A8%8B%E5%A4%8D%E7%8E%B0.md)。

第 5,000 与 6,500 步在同一组 200 道新留出题上均答对 169 道；第 5,000 步 NLL 更低。原有 1,000 题验证集也记录到第二轮 NLL 上升。这些是保留第 5,000 步作为试玩默认权重的依据，不能据此推断对所有任务分布都更好。

## 评测

| checkpoint | 同来源新留出题 | NLL | 自拟题 |
|---|---:|---:|---:|
| 2 万题第 1,250 步 | 162 / 200 | 0.4660 | 14 / 20 |
| 2 万题第 2,500 步 | 166 / 200 | 0.5893 | 16 / 20 |
| **8 万题第 5,000 步** | **169 / 200** | **0.4418** | **20 / 20** |
| 8 万题第 6,500 步 | 169 / 200 | 0.5313 | 20 / 20 |

留出题来自 BoolQ、CommonsenseQA、ARC、C3、OCNLI 的其他样本，每来源 40 题，并排除了本次训练集与原有验证集的题目指纹。自拟题规模较小且较简单。训练还使用了部分来源名为 test 的分片，因此这里的数字不作为对应官方基准测试成绩。完整分组与逐题对照见 [评测报告](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/result/evaluation-2026-09-25/REPORT.md)。

## 本机推理性能

在 RTX 4060 Laptop GPU（8 GiB）上，第 5,000 步权重以 BF16、每批 2 个候选分支、最大 1,024 token 完成 200 道留出题，用时 35.99 秒，约每秒 5.6 题。计时对应模型加载后的固定题集推理；单题耗时会随输入长度、候选数和硬件状态变化。原始聚合记录见 [4060 可行性数据](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/result/analysis/4060_feasibility.json)。

## 局限与用途

- 适用于给定候选项的单项判断和候选排序。试玩中的前 N 项选择是排序后处理，未经过多正确项训练。
- 概率只在当前候选集合内归一化，没有完成跨场景校准。候选措辞、顺序和数量都可能影响结果。
- 评分头输出标量；脚本不提供自由文本回答、拒答判断或通用聊天接口。
- 当前实现分别计算候选分支，没有公共前缀缓存。长文本与跨来源任务尚缺少系统评测。

## 文件与来源

~~~text
models/Qwen3.5-2B-Text/                           文本骨干、配置与分词器
result/unpacked/epoch-00-step-005000/adapter/       LoRA
result/unpacked/epoch-00-step-005000/score_head.pt  评分头
jev/play.py                                        交互与单次推理
jev/common.py、jev/train_cluster.py                提示构造与评分头定义
docs/figures/                                      架构图 PNG 与 SVG
~~~

推理脚本显式加载文本骨干、LoRA 和评分头。本仓库不包含原始题库、转换后的训练题、优化器状态或逐题留出题文本。

## 来源与许可

文本骨干来自 [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，上游模型标注 Apache-2.0。公开代码和权重按本仓库所列许可发布。训练来源还包括 BoolQ、AI2 ARC、CommonsenseQA、MMLU、OCNLI、C3、C-Eval 和 CMMLU；各来源的数据使用条件需要分别遵守。本仓库不再分发这些训练题。

其中 [C-Eval 官方数据集](https://huggingface.co/datasets/ceval/ceval-exam)标注 CC BY-NC-SA 4.0。CMMLU 的使用条件及其他来源详见 [数据来源说明](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/docs/%E6%A8%A1%E5%9E%8B%E5%8D%A1.md)。本仓库的许可声明不改变第三方数据的条款；计划商业使用时，应由使用者逐项核对来源许可。

## 项目

- [GitHub 仓库](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer)
- [模型卡与发布说明](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/docs/%E6%A8%A1%E5%9E%8B%E5%8D%A1.md)

## English guide

### Model and input

This release packages the text backbone extracted from Qwen/Qwen3.5-2B, the LoRA adapter at step 5,000 of the 80,000-question phase, a scalar score head, the tokenizer, and inference code. The text backbone has 1,881,825,088 parameters, 24 decoder layers, and a hidden size of 2,048. Eighteen layers use linear attention and six use full attention.

Each candidate is evaluated with a separate plain-text prompt containing the question, the complete option list, and the option being assessed:

~~~text
Question: {question}
All options:
A. {option A}
B. {option B}
...
Candidate to assess: A. {option A}
Judgment:
~~~

The model uses the final valid token's hidden state and a shared linear head to produce one score per option. Softmax is applied within the supplied option set. Candidate branches share model parameters; the current implementation computes each branch separately and does not cache a common prefix. The CLI accepts 2 to 5 options and rejects branches over 1,024 tokens.

### Quick start

Use a Linux machine with a compatible NVIDIA CUDA GPU and a Python environment that can install the pinned dependencies:

~~~bash
hf download mx-2026/qwen3.5-2b-text-candidate-scorer \
  --local-dir qwen3.5-2b-text-candidate-scorer
cd qwen3.5-2b-text-candidate-scorer
bash jev/setup_env.sh
jev/.venv/bin/python jev/play.py
~~~

Enter one question and 2 to 5 options. Press Enter on the next blank option line, then enter how many of the highest-scoring options to select. Press Enter to use the default of one. A one-shot example is:

~~~bash
jev/.venv/bin/python jev/play.py \
  --question "What is 18 plus 27?" \
  --candidate 46 --candidate 45 --candidate 44 \
  --language en --top-k 2
~~~

The displayed probabilities are relative to the options in the current question. Top N is a ranking operation over scores from a single-answer model, not a separately trained multi-label prediction. Per-question timing includes tokenization and inference; it excludes typing and initial model loading.

### Training and checkpoint selection

Training began with 20,000 bilingual questions for two epochs, reaching 2,500 optimizer steps. The second phase kept those questions and expanded the training set to 80,000; its first epoch ended at step 5,000. Training later paused at step 6,500. This repository publishes the step-5,000 inference weights. The step-5,000 and step-6,500 checkpoints each answered 169 of the same 200 held-out questions correctly, while step 5,000 had lower negative log-likelihood.

| Checkpoint | Held-out correct | NLL | Hand-written correct |
|---|---:|---:|---:|
| 20k, step 1,250 | 162 / 200 | 0.4660 | 14 / 20 |
| 20k, step 2,500 | 166 / 200 | 0.5893 | 16 / 20 |
| **80k, step 5,000** | **169 / 200** | **0.4418** | **20 / 20** |
| 80k, step 6,500 | 169 / 200 | 0.5313 | 20 / 20 |

The 200 held-out questions come from other samples of five training sources, with fingerprint-based exclusion against the training and original validation sets. They do not establish performance on new data sources. The 20 hand-written questions are small and relatively simple. Some source splits named test were used for training, so the figures above are not official benchmark scores. See the [evaluation report](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/result/evaluation-2026-09-25/REPORT.md).

### Local inference measurement

On an RTX 4060 Laptop GPU with 8 GiB of memory, the step-5,000 checkpoint processed 200 held-out questions in 35.99 seconds after loading, about 5.6 questions per second. The run used BF16, a 1,024-token branch limit, and batches of two candidate branches. Input length, option count, hardware, and installation can change latency. The [measurement record](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/result/analysis/4060_feasibility.json) contains the original aggregate.

### Limits, files, and licenses

This model supports supplied-option scoring and ranking. It has not been systematically evaluated for multiple valid answers, rejection, long inputs, or cross-source generalization. It does not provide chat or free-form generation. The repository includes the text backbone, LoRA, score head, tokenizer, diagrams, and CLI. It excludes raw questions, the processed training set, optimizer state, and per-question held-out text.

The backbone originates from [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), whose upstream model card lists Apache-2.0. Training data sources have their own terms. The [official C-Eval dataset](https://huggingface.co/datasets/ceval/ceval-exam) lists CC BY-NC-SA 4.0; check the other source conditions in the [project model card](https://github.com/moX1-2/qwen3.5-2b-text-candidate-scorer/blob/main/docs/%E6%A8%A1%E5%9E%8B%E5%8D%A1.md). This repository's license metadata does not change third-party dataset terms.
