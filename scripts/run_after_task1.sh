#!/bin/bash
# runs after the task 1 evaluation finishes: a second eval pass for any missing benchmark,
# then task 1 qualitative generations, then the task 3 live run. one gpu job at a time on this mac (mps)
cd "$(dirname "$0")/.."
PY=$PWD/.venv/bin/python
export HF_HOME=$PWD/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
A=$PWD/artifacts/task1
LOG=$PWD/artifacts/queue_after_task1.log
say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

say "waiting for the task 1 evaluation"
while pgrep -f "run_task1_eval_local.sh" >/dev/null; do sleep 60; done

# second pass, finished benchmarks are skipped
for v in baseline full_mid full_final l2sp_final lora4_final lora16_final oplora16_final; do
  say "eval pass 2 $v"
  (cd task1 && $PY eval_task1.py --variants "$A/eval_variants.json" --out "$A/eval" --device mps \
    --gsm8k_batch 16 --batch_size 32 --only "$v") > "$A/eval/logs/${v}_pass2.log" 2>&1
  rc=$?
  say "eval pass 2 $v exit $rc"
done
say "results files: $(ls $A/eval/*/*_results.json | wc -l) of 28"

say "starting qualitative"
(cd task1 && $PY qualitative_task1.py --prompts ../data/task1/first_plot_questions.yaml \
  --variants ../artifacts/task1/eval_variants.json --out ../artifacts/task1/qualitative --device mps) \
  > artifacts/task1/qualitative.log 2>&1
rc=$?
say "qualitative exit $rc"

HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-3B-Instruct', revision='aa8e72537993ba99e69dfaafa59ed015b17504d1'))" >> "$LOG" 2>&1
say "starting the task 3 live run"
(cd task3 && $PY live.py run --out ../artifacts/task3/live --device mps) > artifacts/task3/live.log 2>&1
rc=$?
say "live exit $rc"
(cd task3 && $PY live.py summarize --out ../artifacts/task3/live --results ../results/task3/live) >> artifacts/task3/live.log 2>&1
rc=$?
say "summarize exit $rc"
say "all done"
