#!/usr/bin/env bash
set -euo pipefail
# Isolated installation check; never modifies the established GPU environment.
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=/mnt/d/foliorecall_cache/stage4_cpu/huggingface
export TMPDIR=/mnt/d/foliorecall_cache/tmp
export UV_CACHE_DIR=/mnt/d/foliorecall_cache/uv
CPU_ENV=/mnt/d/foliorecall_cache/envs/stage4_cpu
mkdir -p "$TMPDIR" "$HF_HOME"
test -f "$CPU_ENV/bin/python" || python3 -m venv "$CPU_ENV"
uv pip install --python "$CPU_ENV/bin/python" --link-mode=copy \
  torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$CPU_ENV/bin/python" --link-mode=copy \
  '.[demo,eval]' build
"$CPU_ENV/bin/python" -m pip check
