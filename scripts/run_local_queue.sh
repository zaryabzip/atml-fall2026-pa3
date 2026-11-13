#!/bin/bash
# sequential queue. one heavy job at a time to stay inside 16 GB unified memory.
cd "$(dirname "$0")/.."
PY=$PWD/.venv/bin/python
export HF_HOME=$PWD/cache/huggingface
echo "[$(date +%H:%M:%S)] start task2 main training"
$PY task2/train_vlm.py --out artifacts/task2/main --device mps && echo "[$(date +%H:%M:%S)] task2 main training done"
$PY task2/eval_vlm.py --run artifacts/task2/main --device mps && echo "[$(date +%H:%M:%S)] task2 eval done"
$PY task2/report_task2.py --run artifacts/task2/main && echo "[$(date +%H:%M:%S)] task2 report done"
cd task1
for cfg in "lora4 4 8" "lora16 16 32"; do
  set -- $cfg
  echo "[$(date +%H:%M:%S)] start task1 $1"
  $PY train_task1.py --method lora --rank $2 --alpha $3 --out ../artifacts/task1/$1 --device mps --backbone_dtype float16 && echo "[$(date +%H:%M:%S)] task1 $1 done"
done
echo "[$(date +%H:%M:%S)] start task1 oplora16"
$PY train_task1.py --method oplora --rank 16 --alpha 32 --protect_k 128 --out ../artifacts/task1/oplora16 --device mps --backbone_dtype float16 && echo "[$(date +%H:%M:%S)] task1 oplora16 done"
echo "[$(date +%H:%M:%S)] queue finished"
