#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=/mnt/d/foliorecall_cache/stage4_cpu/huggingface
export HF_HUB_OFFLINE=1
export TMPDIR=/mnt/d/foliorecall_cache/tmp
DEMO_OUTPUT=${DEMO_OUTPUT:-/mnt/d/myproject/FolioRecall/outputs/stage4/demo-installed}
cd '/mnt/d/foliorecall_cache/stage4_cpu/moved example'
exec /mnt/d/foliorecall_cache/envs/stage4_cpu/bin/foliorecall serve \
  --index . --config configs/baseline.json --query-config configs/student-ml-cpu.json \
  --port 7864 --output "$DEMO_OUTPUT"
