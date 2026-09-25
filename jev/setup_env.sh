#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

torch_index=$(printenv TORCH_INDEX_URL || true)
if command -v uv >/dev/null 2>&1; then
    if [ ! -x .venv/bin/python ]; then
        uv venv --seed .venv
    fi
    if [ -n "$torch_index" ]; then
        uv pip install --python .venv/bin/python --index-url "$torch_index" torch==2.10.0
    else
        uv pip install --python .venv/bin/python torch==2.10.0
    fi
    uv pip install --python .venv/bin/python -r requirements.txt
else
    if [ ! -x .venv/bin/python ]; then
        python3 -m venv .venv
    fi
    .venv/bin/python -m pip install --upgrade pip
    if [ -n "$torch_index" ]; then
        .venv/bin/python -m pip install torch==2.10.0 --index-url "$torch_index"
    else
        .venv/bin/python -m pip install torch==2.10.0
    fi
    .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python install_causal_conv1d.py
.venv/bin/python -c 'import torch, transformers, peft, fla, causal_conv1d; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())'
