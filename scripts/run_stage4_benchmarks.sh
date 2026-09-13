#!/usr/bin/env bash
set -euo pipefail
# Local verified CPU interpreter; override for another installed environment.
CPU_PYTHON=${CPU_PYTHON:-/mnt/d/foliorecall_cache/envs/stage4_cpu/bin/python}
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=${HF_HOME:-/mnt/d/foliorecall_cache/huggingface}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
for task in hr computer_science finance_en industrial pharmaceuticals; do
  "$CPU_PYTHON" scripts/run_stage4_cpu.py "bm25-$task" -- "$CPU_PYTHON" -m foliorecall evaluate-text \
    --config configs/stage4-evaluation.json --data "outputs/stage4/data/$task" --output "outputs/stage4/bm25/$task"
done
for variant in en ml distilled94; do
  case "$variant" in
    en|ml) student_config="configs/student-$variant-cpu.json" ;;
    distilled94) student_config=outputs/stage3/candidate-configs/distilled-best-cpu.json ;;
  esac
  "$CPU_PYTHON" scripts/run_stage4_cpu.py "hr-$variant-cpu" -- "$CPU_PYTHON" -m foliorecall evaluate \
    --config configs/baseline.json --query-config "$student_config" --index indexes/hr-original \
    --data outputs/stage4/data/hr --output "outputs/stage4/hr-students/$variant" --benchmark
done
