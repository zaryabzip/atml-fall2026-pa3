import json

# evaluation-only notebook for the seven task 1 conditions. no training.
# python task1/make_eval_nb.py  (writes task1/kaggle_task1_eval.ipynb)
# the trained models come in as attached notebook outputs (Add Input -> Notebook output).

OUT = "/Users/zippy/Desktop/ATML/PA3/task1/kaggle_task1_eval.ipynb"
CELLS = []


def md(text):
    CELLS.append(("md", text))


def code(text):
    CELLS.append(("code", text))


md("""# PA3 task 1: evaluation of the seven conditions

Evaluation only. No training happens here.

Before running, attach the two training outputs with **Add Input → Notebook output**:
- the Full FT and L2-SP notebook (its output has `full/` and `l2sp/` folders)
- the LoRA and OPLoRA notebook (its output has `task1/merged/` and the per-run folders)

Then replace the two `REPLACE-...` slugs in the setup cell with the folder names Kaggle shows under `/kaggle/input/`.

Settings: Internet on, accelerator GPU T4 x2. Runs the seven conditions one after another on one GPU. Finished benchmarks are skipped on rerun.

Outputs go to `/kaggle/working/eval`, and the last cell zips them.""")

md("## Setup")

code(r"""%pip install -q lm_eval==0.4.13""")

code(r"""import os, gc, json, time, re, shutil, zipfile
import importlib.metadata as pkg
import torch
import lm_eval
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3-0.6B'
WORK = '/kaggle/working/eval'
# folders of the attached training outputs. replace the slugs with the names Kaggle shows under /kaggle/input/
FULL_ROOT = '/kaggle/input/REPLACE-full-l2sp-output'   # contains full/ and l2sp/
LORA_ROOT = '/kaggle/input/REPLACE-lora-output/task1'  # contains merged/ and the per-run folders
MAX_NEW = 1024
EVAL_LIMIT = None  # None means the full benchmark. set a small number only for a smoke test
DEVICE = 'cuda'
os.makedirs(WORK, exist_ok=True)

VARIANTS = {
    'baseline': MODEL,
    'full_mid': f'{FULL_ROOT}/full/snapshot_625',
    'full_final': f'{FULL_ROOT}/full/snapshot_1250',
    'l2sp_final': f'{FULL_ROOT}/l2sp/snapshot_1250',
    'lora4_final': f'{LORA_ROOT}/merged/lora4',
    'lora16_final': f'{LORA_ROOT}/merged/lora16',
    'oplora16_final': f'{LORA_ROOT}/merged/oplora16',
}
BENCH = [  # name, lm_eval task, chat template, primary metric key
    ('gsm8k', ['gsm8k_cot'], True, 'exact_match,flexible-extract'),
    ('hellaswag', ['hellaswag'], False, 'acc_norm,none'),
    ('boolq', ['boolq'], False, 'acc,none'),
    ('arc_easy', ['arc_easy'], False, 'acc_norm,none'),
]
tok = AutoTokenizer.from_pretrained(MODEL)
print('lm_eval', pkg.version('lm_eval'), '| cuda', torch.cuda.is_available())
for vname, path in VARIANTS.items():
    print(vname, 'found' if os.path.exists(path) else 'MISSING', path)""")

md("## Evaluation\nEach variant runs the four benchmarks with lm_eval 0.4.13. Settings: GSM8K 0-shot with the non-thinking chat template, greedy, 1,024 new tokens. The other three benchmarks use no chat template. Per-example outputs are written before the aggregate results. A GSM8K output is flagged as truncated when it reaches the token cap.")

code(r"""EVAL_FILE = f'{WORK}/eval_results.json'
eval_out = json.load(open(EVAL_FILE)) if os.path.exists(EVAL_FILE) else {}
os.makedirs(f'{WORK}/cache', exist_ok=True)
json.dump({'max_gen_toks': MAX_NEW, 'limit': EVAL_LIMIT, 'lm_eval': pkg.version('lm_eval'),
           'benchmarks': [b[0] for b in BENCH], 'variants': VARIANTS},
          open(f'{WORK}/eval_config.json', 'w'), indent=2)

for vname, path in VARIANTS.items():
    if path.startswith('/') and not os.path.exists(path):
        print('missing model, skipping', vname, path)
        continue
    eval_out.setdefault(vname, {})
    for bname, tasks, chat, key in BENCH:
        if bname in eval_out[vname] and os.path.exists(f'{WORK}/{vname}_{bname}_samples.jsonl'):
            continue
        t0 = time.time()
        model_args = f'pretrained={path},dtype=float16' + (',enable_thinking=False' if chat else '')
        res = lm_eval.simple_evaluate(
            model='hf', model_args=model_args, tasks=tasks, num_fewshot=0,
            batch_size=32 if chat else 8, device=DEVICE, apply_chat_template=chat,
            gen_kwargs={'max_gen_toks': MAX_NEW, 'do_sample': False} if chat else None,
            log_samples=True, limit=EVAL_LIMIT, use_cache=f'{WORK}/cache/{vname}_{bname}')
        samples = res.get('samples', {}).get(tasks[0], [])
        # per-example file first, so an interrupted run keeps what it finished
        with open(f'{WORK}/{vname}_{bname}_samples.jsonl', 'w') as f:
            for s in samples:
                response = s['resps'][0][0] if s.get('resps') else ''
                row = {'doc_id': s.get('doc_id'), 'filter': s.get('filter'), 'target': s.get('target'),
                       'prompt': s.get('arguments'), 'response': response, 'filtered': s.get('filtered_resps')}
                if chat:
                    row['truncated'] = len(tok(response, add_special_tokens=False).input_ids) >= MAX_NEW - 16
                f.write(json.dumps(row, default=str) + '\n')
        metrics = res['results'][tasks[0]]
        eval_out[vname][bname] = metrics
        json.dump(eval_out, open(EVAL_FILE, 'w'), indent=2, default=str)
        print(vname, bname, round(metrics[key], 4), f'{time.time() - t0:.0f} s', flush=True)
        del res
        gc.collect()
        torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
print('evaluation done for:', sorted(eval_out))""")

md("## Zip\nOne zip with the per-example outputs, the aggregate results and the configs.")

code(r"""readme = '''task 1 evaluation outputs

eval_results.json                       aggregate metrics per variant and benchmark
eval_config.json                        settings and the variant paths
<variant>_<benchmark>_samples.jsonl     per example: prompt, response, filtered answer, target, truncation flag (gsm8k)
cache/                                  lm_eval response cache, used to resume a stopped run
'''
open(f'{WORK}/README.txt', 'w').write(readme)
ZIP_PATH = '/kaggle/working/task1_eval_outputs.zip'
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, names in os.walk(WORK):
        for n in names:
            p = os.path.join(root, n)
            z.write(p, os.path.relpath(p, '/kaggle/working'))
print('zip', ZIP_PATH, round(os.path.getsize(ZIP_PATH) / 1e6, 1), 'MB')""")

nb_cells = []
for kind, text in CELLS:
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == "md":
        nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
    else:
        nb_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": nb_cells}
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("cells", len(nb_cells), "->", OUT)
