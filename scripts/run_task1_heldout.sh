#!/bin/bash
# held-out metamathqa evaluation on this mac (mps). one model per process, finished rows are skipped on rerun
cd "$(dirname "$0")/.."
PY=$PWD/.venv/bin/python
export HF_HOME=$PWD/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
LOG=$PWD/artifacts/task1/heldout/logs
mkdir -p "$LOG"
for v in baseline full_mid full_final l2sp_final lora4_final lora16_final oplora16_final; do
  echo "[$(date +%H:%M:%S)] start $v"
  (cd task1 && $PY heldout_task1.py run --only "$v" --device mps) >> "$LOG/$v.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] $v exit $rc"
done
(cd task1 && $PY heldout_task1.py report) >> "$LOG/report.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] report exit $rc"
echo "[$(date +%H:%M:%S)] all done"
