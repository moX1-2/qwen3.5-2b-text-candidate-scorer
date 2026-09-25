"""Install the causal-conv1d wheel matching the active CUDA PyTorch build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path


RELEASE = "v1.7.0"
API = f"https://api.github.com/repos/Dao-AILab/causal-conv1d/releases/tags/{RELEASE}"
CHUNK_SIZE = 8 * 1024 * 1024


def selected_asset() -> dict:
    import torch

    torch_version = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    if torch_version != "2.10":
        raise RuntimeError(f"Expected PyTorch 2.10, found {torch.__version__}")
    if torch.version.cuda is None:
        raise RuntimeError("The installed PyTorch build has no CUDA support")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This installer supports Linux x86_64 wheels only")
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cuda_tag = f"cu{torch.version.cuda.split('.')[0]}"
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    expected = (
        f"causal_conv1d-1.7.0+{cuda_tag}torch{torch_version}"
        f"cxx11abi{abi}-{python_tag}-{python_tag}-linux_x86_64.whl"
    )
    request = urllib.request.Request(API, headers={"User-Agent": "jev-setup"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    matches = [asset for asset in release["assets"] if asset["name"] == expected]
    if len(matches) != 1:
        raise RuntimeError(
            f"No official wheel named {expected}. "
            "Set JEV_CAUSAL_WHEEL to a compatible local wheel path."
        )
    return matches[0]


def download(asset: dict) -> Path:
    cache = Path.home() / ".cache" / "jev-causal-conv1d"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / asset["name"]
    if destination.is_file() and destination.stat().st_size == asset["size"]:
        return destination
    temporary = destination.with_suffix(".part")
    request = urllib.request.Request(
        asset["browser_download_url"], headers={"User-Agent": "jev-setup"}
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            while block := response.read(CHUNK_SIZE):
                output.write(block)
        if temporary.stat().st_size != asset["size"]:
            raise IOError("Downloaded wheel size differs from the GitHub release asset")
        digest = asset.get("digest")
        if digest and digest.startswith("sha256:"):
            sha = hashlib.sha256()
            with temporary.open("rb") as source:
                for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                    sha.update(block)
            if sha.hexdigest() != digest.split(":", 1)[1]:
                raise IOError("Downloaded wheel SHA-256 differs from the release asset")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    override = os.environ.get("JEV_CAUSAL_WHEEL")
    if override:
        wheel = Path(override).expanduser().resolve()
        if not wheel.is_file():
            raise FileNotFoundError(wheel)
        print(f"Using local wheel: {wheel}")
    else:
        asset = selected_asset()
        print(f"Selected official wheel: {asset['name']}", flush=True)
        if args.dry_run:
            return
        wheel = download(asset)
    if args.dry_run:
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)], check=True
    )
    import causal_conv1d

    print(f"causal_conv1d ready: {causal_conv1d.__file__}")


if __name__ == "__main__":
    main()
