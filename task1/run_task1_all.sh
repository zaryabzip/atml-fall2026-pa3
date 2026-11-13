#!/bin/bash
# task 1 pipeline in order. full and l2sp run in parallel, one t4 each. lora and oplora run on mps.
# eval and diagnostics run on cuda
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
RUNS=${RUNS:-../artifacts/task1}
PROMPTS=${PROMPTS:-first_plot_questions.yaml}
DEV=${DEV:-mps}

# step 1 training
REC=${REC:-$RUNS}
CUDA_VISIBLE_DEVICES=0 $PY train_task1.py --method full --out "$RUNS/full" --recovery_dir "$REC/full" &
CUDA_VISIBLE_DEVICES=1 $PY train_task1.py --method l2sp --out "$RUNS/l2sp" --recovery_dir "$REC/l2sp" &
wait
$PY train_task1.py --method lora --rank 4 --alpha 8 --out "$RUNS/lora4" --device "$DEV"
$PY train_task1.py --method lora --rank 16 --alpha 32 --out "$RUNS/lora16" --device "$DEV"
$PY train_task1.py --method oplora --rank 16 --alpha 32 --protect_k 128 --out "$RUNS/oplora16" --device "$DEV"

# step 2 merge adapters into fresh copies with the logit check
for v in lora4 lora16 oplora16; do
  $PY merge_adapters.py --adapters "$RUNS/$v/snapshot_1250/adapters.pt" --out "$RUNS/merged/${v}_final"
done

# step 3 variant map
cat > "$RUNS/variants.json" <<JSON
{
  "baseline": "Qwen/Qwen3-0.6B",
  "full_mid": "$RUNS/full/snapshot_625",
  "full_final": "$RUNS/full/snapshot_1250",
  "l2sp_final": "$RUNS/l2sp/snapshot_1250",
  "lora4_final": "$RUNS/merged/lora4_final",
  "lora16_final": "$RUNS/merged/lora16_final",
  "oplora16_final": "$RUNS/merged/oplora16_final"
}
JSON

# step 4 eval diagnostics qualitative
$PY eval_task1.py --variants "$RUNS/variants.json" --out "$RUNS/eval" --device "${EVAL_DEV:-cuda}" --dtype float16 --batch_size 8
$PY diagnostics_task1.py --full_mid "$RUNS/full/snapshot_625" --full_final "$RUNS/full/snapshot_1250" \
  --l2sp_final "$RUNS/l2sp/snapshot_1250" --lora4 "$RUNS/lora4/snapshot_1250/adapters.pt" \
  --lora16 "$RUNS/lora16/snapshot_1250/adapters.pt" --oplora16 "$RUNS/oplora16/snapshot_1250/adapters.pt" \
  --device "$DEV" --out "$RUNS/diagnostics"
$PY qualitative_task1.py --prompts "$PROMPTS" --variants "$RUNS/variants.json" --out "$RUNS/qualitative" --device "$DEV"

# step 5 report
cat > "$RUNS/runs.json" <<JSON
{"full": "$RUNS/full", "l2sp": "$RUNS/l2sp", "lora4": "$RUNS/lora4", "lora16": "$RUNS/lora16", "oplora16": "$RUNS/oplora16"}
JSON
$PY report_task1.py --runs "$RUNS/runs.json" --eval "$RUNS/eval" --diag "$RUNS/diagnostics" \
  --qual "$RUNS/qualitative" --merged "$RUNS/merged" --out "$RUNS/report"
