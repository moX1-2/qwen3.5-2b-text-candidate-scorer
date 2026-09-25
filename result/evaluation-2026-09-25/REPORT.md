# 四个 checkpoint 的固定测试集评测

评测日期：2026-09-25。全部模型使用同一批 200 道留出题和 20 道自拟题。
留出题从五个原始数据源的 validation 分片按固定种子抽取，每源 40 题；已排除 8 万题训练集及原有 1,000 题验证集的题目指纹。
ARC 的 40 题由 Easy 和 Challenge 各 20 题组成。抽样方法与哈希见 `make_test_sets.py`、`test_set_manifest.json`。
推理使用训练时的候选分支提示、1024 token 截断、LoRA 和评分头；不使用语言模型自由生成。

## 总体结果

| checkpoint | 留出题正确数 / 200 | 留出题 NLL | 留出题 Brier | 自拟题正确数 / 20 | 自拟题 NLL |
|---|---:|---:|---:|---:|---:|
| 20k_step1250 | 162 / 200 | 0.4660 | 0.2585 | 14 / 20 | 0.4493 |
| 20k_step2500 | 166 / 200 | 0.5893 | 0.2717 | 16 / 20 | 0.5796 |
| 80k_step5000 | 169 / 200 | 0.4418 | 0.2263 | 20 / 20 | 0.0777 |
| 80k_step6500 | 169 / 200 | 0.5313 | 0.2525 | 20 / 20 | 0.0516 |

## 各来源正确数

| checkpoint | BoolQ / 40 | CommonsenseQA / 40 | ARC / 40 | C3 / 40 | OCNLI / 40 |
|---|---:|---:|---:|---:|---:|
| 20k_step1250 | 38 | 29 | 36 | 36 | 23 |
| 20k_step2500 | 37 | 27 | 36 | 36 | 30 |
| 80k_step5000 | 37 | 28 | 36 | 36 | 32 |
| 80k_step6500 | 39 | 27 | 36 | 36 | 31 |

## 相邻 checkpoint 的逐题变化

| 测试集 | 对比 | 新答对 | 新答错 | 平均 NLL 变化 |
|---|---|---:|---:|---:|
| heldout_200 | 20k_step1250 -> 20k_step2500 | 13 | 9 | +0.1232 |
| heldout_200 | 20k_step2500 -> 80k_step5000 | 6 | 3 | -0.1475 |
| heldout_200 | 80k_step5000 -> 80k_step6500 | 2 | 2 | +0.0896 |
| custom_20 | 20k_step1250 -> 20k_step2500 | 2 | 0 | +0.1303 |
| custom_20 | 20k_step2500 -> 80k_step5000 | 4 | 0 | -0.5018 |
| custom_20 | 80k_step5000 -> 80k_step6500 | 0 | 0 | -0.0262 |

## 自拟题逐题结果

候选项顺序由固定种子打乱，表中写出模型选中的候选文本。

| 题号 | 题目 | 标准答案 | 1250 步 | 2500 步 | 5000 步 | 6500 步 |
|---|---|---|---|---|---|---|
| custom:001 | 18 加 27 等于多少？ | 45 | ✓ 45 | ✓ 45 | ✓ 45 | ✓ 45 |
| custom:002 | 一本书 80 页，已经读了 35 页，还剩多少页？ | 45 页 | ✗ 55 页 | ✗ 55 页 | ✓ 45 页 | ✓ 45 页 |
| custom:003 | 今天是星期三，两天后是星期几？ | 星期五 | ✗ 星期四 | ✗ 星期四 | ✓ 星期五 | ✓ 星期五 |
| custom:004 | 甲比乙高，乙比丙高。三人中谁最高？ | 甲 | ✓ 甲 | ✓ 甲 | ✓ 甲 | ✓ 甲 |
| custom:005 | 小王把蓝色杯子放在书架上，把红色杯子放在桌上。哪个杯子在书架上？ | 蓝色杯子 | ✓ 蓝色杯子 | ✓ 蓝色杯子 | ✓ 蓝色杯子 | ✓ 蓝色杯子 |
| custom:006 | 一小时有多少分钟？ | 60 分钟 | ✓ 60 分钟 | ✓ 60 分钟 | ✓ 60 分钟 | ✓ 60 分钟 |
| custom:007 | 三张 20 元纸币一共是多少元？ | 60 元 | ✗ 80 元 | ✓ 60 元 | ✓ 60 元 | ✓ 60 元 |
| custom:008 | 盒子里只有苹果和梨。取出一个水果，确认它不是苹果。它是什么？ | 梨 | ✓ 梨 | ✓ 梨 | ✓ 梨 | ✓ 梨 |
| custom:009 | 通知写着会议 14:00 开始，13:45 签到。签到应在几点？ | 13:45 | ✓ 13:45 | ✓ 13:45 | ✓ 13:45 | ✓ 13:45 |
| custom:010 | 一件商品原价 50 元，减价 10 元后售价多少？ | 40 元 | ✗ 45 元 | ✗ 45 元 | ✓ 40 元 | ✓ 40 元 |
| custom:011 | What is 7 multiplied by 8? | 56 | ✓ 56 | ✓ 56 | ✓ 56 | ✓ 56 |
| custom:012 | If today is Monday, what day is three days later? | Thursday | ✗ Wednesday | ✗ Wednesday | ✓ Thursday | ✓ Thursday |
| custom:013 | Maya put the keys in the drawer and the wallet on the table. Where are the keys? | In the drawer | ✓ In the drawer | ✓ In the drawer | ✓ In the drawer | ✓ In the drawer |
| custom:014 | How many centimeters are in one meter? | 100 | ✓ 100 | ✓ 100 | ✓ 100 | ✓ 100 |
| custom:015 | All squares have four sides. A shape is a square. How many sides does it have? | Four | ✓ Four | ✓ Four | ✓ Four | ✓ Four |
| custom:016 | A train leaves at 9:00 and arrives at 11:30. How long is the trip? | 2 hours 30 minutes | ✗ 3 hours 30 minutes | ✓ 2 hours 30 minutes | ✓ 2 hours 30 minutes | ✓ 2 hours 30 minutes |
| custom:017 | The note says the library closes at 6 p.m. When does it close? | 6 p.m. | ✓ 6 p.m. | ✓ 6 p.m. | ✓ 6 p.m. | ✓ 6 p.m. |
| custom:018 | Lena is older than Sam, and Sam is older than Kai. Who is youngest? | Kai | ✓ Kai | ✓ Kai | ✓ Kai | ✓ Kai |
| custom:019 | A box contains 12 pencils. Four are removed. How many remain? | 8 | ✓ 8 | ✓ 8 | ✓ 8 | ✓ 8 |
| custom:020 | Which tool is normally used to cut a sheet of paper? | Scissors | ✓ Scissors | ✓ Scissors | ✓ Scissors | ✓ Scissors |

## 解读边界

- 200 道留出题来自五个已有训练来源的其他样本，检验同来源新题；这不是跨来源泛化测试。
- 20 道自拟题由人工设计，题目较简单，主要展示实际候选打分输出，不用于精确估计总体正确率。
- 第 5,000 与 6,500 步在留出题上正确数相同，后者 NLL 更高。该变化与训练日志中原有 1,000 题验证集的 NLL 走势一致。
- 原始逐题概率保存在 `run/` 与 `run_1250/` 中，可检查错误集合和高置信度错误。
