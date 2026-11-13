#!/bin/bash
# task 2 pipeline in order. prep and feature cache are already done by default.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
RUN=${RUN:-../artifacts/task2/main}
DEV=${DEV:-mps}

if [ "${REDO_PREP:-0}" = "1" ]; then
  $PY prep_data.py
fi
if [ ! -f ../data/task2/features/cache_meta.json ]; then
  $PY cache_features.py "$DEV"
fi

$PY train_vlm.py --out "$RUN" --device "$DEV"
$PY eval_vlm.py --run "$RUN" --device "$DEV"
$PY report_task2.py --run "$RUN"
