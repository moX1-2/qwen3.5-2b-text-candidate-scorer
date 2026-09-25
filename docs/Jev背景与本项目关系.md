# 官方 Jev 的背景与本项目关系

核对日期：2026-09-25。以下关于官方 Jev 的描述依据 TypeSafe AI 的[发布公告](https://typesafe.ai/blog/introducing-system-one-models-and-jev)与[开发文档](https://docs.typesafe.ai/introduction)；本项目的实现范围依据 `jev/common.py`、`jev/train_cluster.py` 和本地评测记录。

## 官方 Jev 是什么

TypeSafe AI 于 2026-09-15 发布 Jev 的早期访问版本，将其称为首个 **System One** 模型。这个名称借用了 Daniel Kahneman 对快速判断的区分；“Jev”取自 William Stanley Jevons。TypeSafe 将产品定位为在软件工作流中回答范围明确的问题：应用提供文本状态和预先定义的问题，模型返回可供代码使用的类型化结果与概率，不生成供人阅读的自由文本。见[官方公告](https://typesafe.ai/blog/introducing-system-one-models-and-jev)与[System One 说明](https://docs.typesafe.ai/concepts/system-one)。

官方接口目前有三类问题：[Choice](https://docs.typesafe.ai/primitives/choice) 从给定选项中选一个并返回各项概率；[Score](https://docs.typesafe.ai/primitives/score) 按有序等级给出分数与等级概率；[Noul](https://docs.typesafe.ai/primitives/noul) 返回一个是非命题为真的概率。一次请求可以对同一状态提出多个相互独立的问题。Choice 和 Score 另有根据概率分布形状计算的 `confidence` 字段，Noul 没有这个独立字段。官方文档明确指出，`confidence` 高表示分布更集中，不能保证单次判断正确。见[接口概览](https://docs.typesafe.ai/introduction)与[置信度说明](https://docs.typesafe.ai/confidence)。

TypeSafe 在公告中将其训练方法称为 **Reinforcement Learning for Calibrated Decisions（RLCD）**，并介绍并行采样与概率校准目标。这些是厂商对官方模型的技术描述。所查官方页面没有给出可据以重建权重的完整架构、训练数据和训练配方，本项目也没有使用官方权重或 RLCD。官方速度与成本数据是其服务端场景的发布方结果，不能用于推断本地实验的性能。见[发布公告](https://typesafe.ai/blog/introducing-system-one-models-and-jev)。

## 本项目实现了什么

本仓库的 JEV 是**受上述决策接口启发的独立候选判断实验**，与 TypeSafe AI 无关联。它以 Qwen3.5-2B 的文本骨干、LoRA 和线性评分头为基础，对每个候选项分别构造提示与计算分数，再在同一题内使用 Softmax 生成概率，以单正确项标签和交叉熵训练。当前训练样本有 2 至 5 个候选项，训练与评测只覆盖单题单选。模型卡和全流程记录分别见 `docs/模型卡.md`、`docs/全流程复现.md`。

| 维度 | TypeSafe 官方 Jev | 本项目 JEV |
|---|---|---|
| 输入与输出 | 文本状态加类型化问题，返回结构化判断 | 问题描述加候选列表，返回组内选项分数与概率 |
| 已覆盖问题类型 | Choice、Score、Noul | 单正确项 Choice 风格判断 |
| 多问题处理 | 官方文档描述同一请求内并行、独立评估 | 每道题的候选分支分别前向计算 |
| 训练方法 | 发布方称为 RLCD | 有监督交叉熵，AdamW 更新 LoRA 与评分头 |
| 概率与置信度 | Choice/Score 含概率分布与独立的 `confidence` 字段 | 输出 Softmax 概率；未实现官方 `confidence` 定义 |
| 模型来源 | TypeSafe 的官方托管模型 | Qwen3.5 文本骨干上的本地训练 checkpoint |

本项目尚未评测官方 Jev，也没有可直接比较的同题、同接口、同硬件速度或质量结果。现有 200 道新留出题与 20 道自拟题只用于比较本项目四个 checkpoint。文档与仓库名称中的“JEV”表示本地实验代号，不表示官方实现、授权或合作。
