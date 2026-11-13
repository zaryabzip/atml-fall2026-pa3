#!/bin/bash
# task 1 evaluation on this mac (mps). one model per process in a fixed order
# finished benchmarks are skipped on rerun so a stopped run can be restarted
cd "$(dirname "$0")/.."
PY=$PWD/.venv/bin/python
export HF_HOME=$PWD/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
A=$PWD/artifacts/task1
VARIANTS=${VARIANTS:-$A/eval_variants.json}
OUT=${OUT:-$A/eval}
mkdir -p "$OUT/logs"
for v in baseline full_mid full_final l2sp_final lora4_final lora16_final oplora16_final; do
  echo "[$(date +%H:%M:%S)] start $v"
  (cd task1 && $PY eval_task1.py --variants "$VARIANTS" --out "$OUT" --device mps --gsm8k_batch 16 --batch_size 32 --only "$v") >> "$OUT/logs/$v.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] $v exit $rc"
done
echo "[$(date +%H:%M:%S)] all done"
