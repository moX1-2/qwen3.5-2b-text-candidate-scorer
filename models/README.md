# 基座模型目录

本实验使用 [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 的文本骨干。上游提交为 `15852e8c16360a2fea060d615a32b45270f8a8fc`，本地提取脚本为 `jev/convert_text_model.py`。

将上游模型下载到 `models/Qwen3.5-2B/` 后，从仓库根目录运行：

```bash
python3 jev/convert_text_model.py \
  --source models/Qwen3.5-2B \
  --destination models/Qwen3.5-2B-Text
```

提取过程移除视觉与 MTP 张量，保留 320 个文本张量。`Qwen3.5-2B-Text/model.safetensors` 大小为 3,763,691,928 字节，SHA-256 为 `6c2df657da48151eafd02f33c1cc1f0dd1524043cb67b368fdac6d09e98f6851`。公开仓库与 checkpoint 压缩包都不提供该文件。完整来源与复现步骤见 `docs/全流程复现.md`。
