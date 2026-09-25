"""Create a text-only Qwen3.5 checkpoint from the local multimodal checkpoint.

The safetensors payload is copied in chunks. The 4.5 GB source checkpoint is
never loaded into RAM as a tensor dictionary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path


NEW = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = NEW / "models" / "Qwen3.5-2B"
DEFAULT_DESTINATION = NEW / "models" / "Qwen3.5-2B-Text"
PREFIX = "model.language_model."
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
)
COPY_CHUNK = 8 * 1024 * 1024


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        header = json.loads(handle.read(header_length))
    return header, 8 + header_length


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(source: Path) -> str | None:
    readme = source.parent / "README.md"
    if not readme.is_file():
        return None
    match = re.search(r"仓库提交：[^0-9a-f]*([0-9a-f]{40})", readme.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def convert(source: Path, destination: Path, force: bool) -> None:
    weights = source / "model.safetensors-00001-of-00001.safetensors"
    source_config = source / "config.json"
    if not weights.is_file() or not source_config.is_file():
        raise FileNotFoundError("Source model weights or config.json are missing")

    target_weights = destination / "model.safetensors"
    if target_weights.exists() and not force:
        raise FileExistsError(f"{target_weights} exists; pass --force to replace it")

    original, data_start = read_header(weights)
    selected = []
    for name, spec in original.items():
        if name.startswith(PREFIX):
            offsets = spec["data_offsets"]
            if len(offsets) != 2 or offsets[1] <= offsets[0]:
                raise ValueError(f"Invalid offsets for {name}")
            selected.append((offsets[0], offsets[1], name, spec))
    selected.sort()
    if len(selected) < 300:
        raise ValueError(f"Only {len(selected)} text tensors found")
    if not any(name == PREFIX + "embed_tokens.weight" for _, _, name, _ in selected):
        raise ValueError("Text token embedding is missing")

    parent_config = json.loads(source_config.read_text(encoding="utf-8"))
    text_config = dict(parent_config["text_config"])
    text_config["model_type"] = "qwen3_5_text"
    text_config["architectures"] = ["Qwen3_5ForCausalLM"]
    text_config["tie_word_embeddings"] = True

    new_header: dict[str, dict] = {}
    payload_size = 0
    parameter_count = 0
    for start, end, name, spec in selected:
        new_name = "model." + name[len(PREFIX) :]
        if new_name in new_header:
            raise ValueError(f"Duplicate target tensor: {new_name}")
        new_header[new_name] = {
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "data_offsets": [payload_size, payload_size + end - start],
        }
        payload_size += end - start
        parameter_count += math.prod(spec["shape"])

    header_bytes = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "model.safetensors.tmp"
    try:
        with weights.open("rb") as src, temporary.open("wb") as dst:
            dst.write(struct.pack("<Q", len(header_bytes)))
            dst.write(header_bytes)
            for start, end, _, _ in selected:
                src.seek(data_start + start)
                remaining = end - start
                while remaining:
                    block = src.read(min(COPY_CHUNK, remaining))
                    if not block:
                        raise EOFError("Source checkpoint ended during conversion")
                    dst.write(block)
                    remaining -= len(block)
        if temporary.stat().st_size != 8 + len(header_bytes) + payload_size:
            raise IOError("Converted checkpoint size does not match its header")
        os.replace(temporary, target_weights)
    finally:
        temporary.unlink(missing_ok=True)

    (destination / "config.json").write_text(
        json.dumps(text_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for filename in TOKENIZER_FILES + ("LICENSE",):
        path = source / filename
        if path.is_file():
            shutil.copy2(path, destination / filename)

    converted, converted_start = read_header(target_weights)
    if len(converted) != len(selected):
        raise IOError("Converted tensor count is incorrect")
    if any("visual" in key or key.startswith("mtp.") for key in converted):
        raise IOError("Non-text tensor found in converted checkpoint")
    if max(spec["data_offsets"][1] for spec in converted.values()) != payload_size:
        raise IOError("Converted payload offsets are incorrect")
    if converted_start + payload_size != target_weights.stat().st_size:
        raise IOError("Converted payload is incomplete")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": str(source),
        "source_revision": source_revision(source),
        "source_weight_sha256": sha256(weights),
        "text_weight_sha256": sha256(target_weights),
        "text_tensors": len(selected),
        "text_parameters": parameter_count,
        "text_weight_bytes": target_weights.stat().st_size,
        "excluded": ["vision", "mtp"],
    }
    (destination / "conversion.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / ".gitignore").write_text("model.safetensors\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert(args.source.resolve(), args.destination.resolve(), args.force)


if __name__ == "__main__":
    main()
