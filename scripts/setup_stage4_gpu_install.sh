#!/usr/bin/env bash
set -euo pipefail
# Dependency installation only. CUDA remains hidden; no model/check-env invocation.
export CUDA_VISIBLE_DEVICES=""
export TMPDIR=/mnt/d/foliorecall_cache/tmp
export UV_CACHE_DIR=/mnt/d/foliorecall_cache/uv
GPU_ENV=/mnt/d/foliorecall_cache/envs/stage4_gpu_install
mkdir -p "$TMPDIR"
test -f "$GPU_ENV/bin/python" || python3 -m venv "$GPU_ENV"
uv pip install --python "$GPU_ENV/bin/python" --link-mode=copy \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$GPU_ENV/bin/python" --link-mode=copy "$1[image,demo]"
"$GPU_ENV/bin/python" -m pip check
