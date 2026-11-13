import json

# training-only notebook for lora4, lora16 and oplora16. no eval, diagnostics or report.
# python task1/make_lora_nb.py  (writes task1/kaggle_task1_lora_train.ipynb)

OUT = "/Users/zippy/Desktop/ATML/PA3/task1/kaggle_task1_lora_train.ipynb"
CELLS = []


def md(text):
    CELLS.append(("md", text))


def code(text):
    CELLS.append(("code", text))


md("""# PA3 task 1: LoRA-4, LoRA-16 and OPLoRA-16 training

Training only. Evaluation runs later on the saved models, on a local machine.
Settings: Internet on, accelerator GPU T4 x2.

Run the cells in order. Each run trains on its own GPU, in its own thread. LoRA-16 and OPLoRA-16 run together first, then LoRA-4 starts when a GPU is free. The same pairing is used for the cost comparison.

Do not restart the kernel during training. A restart stops the training threads. The recovery files then let the runs resume from their last save (updates 312, 625, 937 or 1250) when the launch cell is run again in the same session folder. Interrupting the launch cell does not stop the runs. Rerun it to wait for them.

Everything is saved under `/kaggle/working/task1`. The last cell writes one zip with the trained adapters, the merged models, the logs and the configs.""")

md("## Setup")

code(r"""import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # set before the tokenizer loads, avoids fork warnings
import gc, sys, json, time, random, re, math, shutil, threading, traceback, zipfile
import importlib.metadata as pkg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
try:
    from IPython.display import clear_output
except ImportError:  # outside a notebook
    def clear_output(wait=False):
        pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL = 'Qwen/Qwen3-0.6B'
WORK = '/kaggle/working/task1'
ZIP_PATH = '/kaggle/working/task1_lora_outputs.zip'
SEED = 6304
NUM_EXAMPLES = 40000
EFFECTIVE_BATCH = 32
MICRO_BATCH = 2
MAX_LEN = 2048
UPDATES = 1250
WARMUP = 38
CE_CHUNK = 512
DROPOUT = 0.05
PROTECT_K = 128
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
RECOVERY_STEPS = {312, 625, 937, UPDATES}  # spec save points. one recovery file per run, overwritten at each save
RUNS = {  # name: (method, rank, alpha, peak lr)
    'lora16': ('lora', 16, 32, 2e-4),
    'oplora16': ('oplora', 16, 32, 2e-4),
    'lora4': ('lora', 4, 8, 2e-4),
}
LOAD_LOCK = threading.Lock()  # loading two models in threads at once failed in testing
RUN_ORDER = ['lora16', 'oplora16', 'lora4']  # the first two run side by side, so their timings are comparable
GPUS = [0, 1]  # one run per GPU at a time
PROGRESS_EVERY = 60  # seconds between progress prints
USE_CUDA = torch.cuda.is_available()  # each training thread sets its own gpu
DEVICE = torch.device('cpu')  # merge and merge checks run on CPU
for d in [WORK, f'{WORK}/merged']:
    os.makedirs(d, exist_ok=True)
print(f'free disk: {shutil.disk_usage(WORK).free / 1e9:.1f} GB | cuda: {USE_CUDA}')
print({m: pkg.version(m) for m in ['torch', 'transformers', 'datasets']})
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)""")

md("## Data\nMetaMathQA rows are filtered as in the spec. GSM8K test questions are removed. The first 40,000 eligible rows are kept, in source order. Their row numbers are saved.")

code(r"""def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

tok = AutoTokenizer.from_pretrained(MODEL)
gsm8k = load_dataset('openai/gsm8k', 'main', split='test')
gsm8k_questions = {normalize(q) for q in gsm8k['question']}

meta_path = hf_hub_download('meta-math/MetaMathQA', 'MetaMathQA-395K.json', repo_type='dataset')
meta_rows = json.load(open(meta_path))

examples, example_row_ids = [], []  # (prompt ids, response ids), and the source row numbers
for row_id, r in enumerate(meta_rows):
    query, response = r['query'], r['response']
    if not query.strip() or not response.strip():
        continue
    if normalize(query) in gsm8k_questions or normalize(r.get('original_question', '')) in gsm8k_questions:
        continue
    prompt = tok.apply_chat_template([{'role': 'user', 'content': query}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(response + '<|im_end|>', add_special_tokens=False).input_ids
    if len(p_ids) + len(a_ids) > MAX_LEN:
        continue
    examples.append((p_ids, a_ids))
    example_row_ids.append(row_id)
    if len(examples) == NUM_EXAMPLES:
        break
del meta_rows
assert len(examples) >= UPDATES * EFFECTIVE_BATCH, 'not enough eligible examples'
json.dump({'source': 'meta-math/MetaMathQA MetaMathQA-395K.json', 'row_ids': example_row_ids,
           'rule': 'empty rows, gsm8k test questions and rows over 2048 tokens removed'},
          open(f'{WORK}/train_examples.json', 'w'))
print('examples kept:', len(examples))""")

md("## Helper functions")

code(r"""def collate(batch, device):
    L = max(len(p) + len(a) for p, a in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for k, (p, a) in enumerate(batch):
        seq = p + a
        ids[k, :len(seq)] = torch.tensor(seq)
        att[k, :len(seq)] = 1
        lab[k, len(p):len(seq)] = torch.tensor(a)
    return ids.to(device), att.to(device), lab.to(device)


def _ce_part(head, h, t):
    return F.cross_entropy(head(h).float(), t, reduction='sum')


def chunked_ce(head, h, t):
    # vocab logits are rebuilt chunk by chunk in backward, so only one chunk is in memory
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], CE_CHUNK):
        total = total + checkpoint(_ce_part, head, h[i:i + CE_CHUNK], t[i:i + CE_CHUNK], use_reentrant=False)
    return total


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)


class LoRALinear(nn.Module):
    # frozen base linear plus a low rank update. with protect_k > 0 the update is projected (oplora).
    def __init__(self, base, rank, alpha, dropout, seed):
        super().__init__()
        self.base = base
        din, dout = base.in_features, base.out_features
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)
        g = torch.Generator().manual_seed(seed)
        A = (torch.rand(rank, din, generator=g) * 2 - 1) / math.sqrt(din)  # same start for lora and oplora
        self.A = nn.Parameter(A.to(base.weight.device))
        self.B = nn.Parameter(torch.zeros(dout, rank, device=base.weight.device))
        self.protect_k = 0
        self.Uk = None
        self.Vk = None

    def set_bases(self, W0, k):
        # leading singular bases of the starting weight, computed in fp32 on cpu
        U, _, Vh = torch.linalg.svd(W0.float().cpu(), full_matrices=False)
        dev = self.A.device
        self.Uk = U[:, :k].contiguous().to(dev)
        self.Vk = Vh[:k].T.contiguous().to(dev)
        self.protect_k = k

    def effective(self):
        A, B = self.A, self.B
        if self.protect_k:
            A = A - (A @ self.Vk) @ self.Vk.T
            B = B - self.Uk @ (self.Uk.T @ B)
        return A, B

    def forward(self, x):
        A, B = self.effective()
        return self.base(x) + self.scale * ((self.drop(x) @ A.T) @ B.T)


def apply_lora(model, rank, alpha, dropout, protect_k):
    # replaces every target linear. returns the seconds spent on the oplora bases
    setup_seconds = 0.0
    idx = 0
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in TARGETS:
                if not hasattr(parent, name):
                    continue
                base = getattr(parent, name)
                idx += 1
                mod = LoRALinear(base, rank, alpha, dropout, seed=SEED * 1000 + idx)
                if protect_k:
                    t0 = time.time()
                    mod.set_bases(base.weight.detach(), protect_k)
                    setup_seconds += time.time() - t0
                setattr(parent, name, mod)
    return setup_seconds


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def export_adapters(model, path, meta):
    # raw and effective factors per target. oplora also keeps its bases and the leakage checks.
    out = {'meta': meta, 'modules': {}, 'bases': {}, 'checks': {}}
    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    with torch.no_grad():
        for name, m in mods:
            A_eff, B_eff = m.effective()
            out['modules'][name] = {'A': m.A.detach().float().cpu(), 'B': m.B.detach().float().cpu(),
                                    'A_eff': A_eff.detach().float().cpu(), 'B_eff': B_eff.detach().float().cpu(),
                                    'scale': m.scale}
            if m.protect_k:
                out['bases'][name] = {'Uk': m.Uk.detach().float().cpu(), 'Vk': m.Vk.detach().float().cpu()}
        if meta['protect_k']:
            out['checks'] = {
                'max_leak_out': max(float((m.Uk.T @ m.effective()[1]).abs().max()) for _, m in mods),
                'max_leak_in': max(float((m.effective()[0] @ m.Vk).abs().max()) for _, m in mods)}
    torch.save(out, path)
    return out['checks']


def save_recovery(rdir, step, named, opt, sched, scaler, device, extra):
    os.makedirs(rdir, exist_ok=True)
    state = {'step': step, 'trainable': {n: p.detach().cpu() for n, p in named},
             'opt': opt.state_dict(), 'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
             'rng_cpu': torch.get_rng_state(), 'extra': extra}
    if device.type == 'cuda':
        state['rng_cuda'] = torch.cuda.get_rng_state(device)
    tmp = f'{rdir}/recovery.tmp'
    torch.save(state, tmp)
    os.replace(tmp, f'{rdir}/recovery.pt')


def load_recovery(rdir, named, opt, sched, scaler, device):
    path = f'{rdir}/recovery.pt'
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    lookup = dict(named)
    with torch.no_grad():
        for n, v in ck['trainable'].items():
            lookup[n].copy_(v.to(device))
    opt.load_state_dict(ck['opt'])
    sched.load_state_dict(ck['sched'])
    scaler.load_state_dict(ck['scaler'])
    torch.set_rng_state(ck['rng_cpu'])
    if device.type == 'cuda' and 'rng_cuda' in ck:
        torch.cuda.set_rng_state(ck['rng_cuda'], device)
    return ck""")

md("## Training function\nOne function for every LoRA run. The backbone is frozen fp16 on a GPU, and the adapters are fp32. Loss is normalized over all response tokens in the effective batch. Gradients are unscaled before clipping. Skipped steps are logged.")

code(r"""def train_run(name, gpu):
    method, rank, alpha, lr = RUNS[name]
    device = torch.device(f'cuda:{gpu}') if USE_CUDA else torch.device('cpu')
    if USE_CUDA:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.manual_seed(SEED)  # this device's generator only, the other run keeps its own
    else:
        torch.manual_seed(SEED)
    out = f'{WORK}/{name}'
    rdir = f'{out}/recovery'
    os.makedirs(out, exist_ok=True)
    protect_k = PROTECT_K if method == 'oplora' else 0

    dtype = torch.float16 if USE_CUDA else torch.float32  # frozen backbone in fp16 on a gpu
    with LOAD_LOCK:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model = model.to(device)
    assert next(model.parameters()).dtype == dtype, 'backbone dtype is not the one asked for'
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    setup_seconds = apply_lora(model, rank, alpha, DROPOUT, protect_k)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    n_trainable = sum(p.numel() for p in params)
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=USE_CUDA)
    sched = get_cosine_schedule_with_warmup(opt, WARMUP, UPDATES)
    scaler = torch.amp.GradScaler(device.type, enabled=USE_CUDA, init_scale=2 ** 11)

    log, start = [], 0
    ck = load_recovery(rdir, named, opt, sched, scaler, device)
    if ck is not None:
        log = ck['extra']['log']
        setup_seconds = ck['extra']['setup_seconds']
        start = ck['step']
        print(name, 'resumed at update', start, flush=True)
        del ck

    log_fh = open(f'{out}/train.log', 'a', buffering=1)  # progress bar goes here, the launch cell shows its last line
    bar = tqdm(total=UPDATES, initial=start, desc=name, file=log_fh)
    for step in range(start, UPDATES):
        t0 = time.time()
        batch = examples[step * EFFECTIVE_BATCH:(step + 1) * EFFECTIVE_BATCH]
        total_tokens = sum(len(a) for _, a in batch)  # loss is normalized over all response tokens
        sft_sum = torch.zeros((), device=device)
        for i in range(0, EFFECTIVE_BATCH, MICRO_BATCH):
            ids, att, lab = collate(batch[i:i + MICRO_BATCH], device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=USE_CUDA):
                hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
                mask = lab[:, 1:] != -100
                mb_loss = chunked_ce(model.lm_head, hidden[:, :-1][mask], lab[:, 1:][mask])
            scaler.scale(mb_loss / total_tokens).backward()
            sft_sum += mb_loss.detach()

        scale_before = scaler.get_scale()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        lr_used = opt.param_groups[0]['lr']
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        opt.zero_grad(set_to_none=True)
        sched.step()

        sft = float(sft_sum) / total_tokens
        rec = {'update': step + 1, 'sft_loss': sft, 'lr': lr_used, 'grad_norm': float(grad_norm),
               'skipped': skipped, 'seconds': round(time.time() - t0, 2)}
        if USE_CUDA:
            rec['peak_mem_GB'] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        log.append(rec)
        bar.update(1)
        bar.set_postfix(sft=f'{sft:.4f}', lr=f'{lr_used:.1e}', s=rec['seconds'])

        done = step + 1
        if done in RECOVERY_STEPS:
            save_recovery(rdir, done, named, opt, sched, scaler, device,
                          {'log': log, 'setup_seconds': setup_seconds})
    bar.close()
    log_fh.close()

    # final adapters: written before the summary, so a finished summary means a complete snapshot
    snap = f'{out}/snapshot_{UPDATES}'
    os.makedirs(snap, exist_ok=True)
    t_snap = time.time()
    meta = {'method': method, 'rank': rank, 'alpha': alpha, 'protect_k': protect_k, 'lr': lr,
            'dropout': DROPOUT, 'targets': TARGETS, 'updates': UPDATES}
    checks = export_adapters(model, f'{snap}/adapters.pt', meta)
    json.dump(checks, open(f'{snap}/leakage.json', 'w'), indent=2)
    tok.save_pretrained(snap)
    snap_seconds = round(time.time() - t_snap, 2)
    # the recovery file at the last save point is kept, so the run can be restarted from it
    json.dump(log, open(f'{out}/train_log.json', 'w'))
    summary = {'run': name, 'method': method, 'rank': rank, 'alpha': alpha, 'lr': lr,
               'updates': UPDATES, 'trainable_params': n_trainable,
               'trainable_bytes_fp32': 4 * n_trainable,
               'train_seconds': round(sum(r['seconds'] for r in log), 1),
               'mean_seconds_per_update': round(sum(r['seconds'] for r in log) / len(log), 2),
               'peak_mem_GB': round(max(r.get('peak_mem_GB', 0) for r in log), 2) if USE_CUDA else None,
               'setup_seconds': round(setup_seconds, 2),
               'basis_bytes': (sum((m.Uk.numel() + m.Vk.numel()) * 4 for m in lora_modules(model))
                               if protect_k else 0),
               'adapter_file_bytes': os.path.getsize(f'{snap}/adapters.pt'),
               'snapshot_dir_bytes': dir_bytes(snap),
               'snapshot_export_seconds': snap_seconds,
               'recovery_file_bytes': os.path.getsize(f'{rdir}/recovery.pt'),
               'skipped_updates': sum(1 for r in log if r['skipped']),
               'final_sft_loss': log[-1]['sft_loss'],
               'backbone_dtype': str(dtype), 'checks': checks, 'gpu': gpu if USE_CUDA else None}
    json.dump(summary, open(f'{out}/summary.json', 'w'), indent=2)
    del model, opt, sched, scaler, params
    gc.collect()
    return summary""")

md("## Launch\nEach run trains in a thread on its own GPU. The progress bars show the update count and the loss. A run that fails writes `error.txt` and the others keep going. This cell waits until every run has finished.\n\nRerunning this cell waits for runs that are still going, and it starts runs that have no summary yet.")

code(r"""LIVE = globals().setdefault('LIVE', {})  # name -> thread, kept across reruns of this cell
GPU_OF = globals().setdefault('GPU_OF', {})  # name -> gpu index


def run_safe(name, gpu):
    try:
        train_run(name, gpu)
    except Exception:
        text = traceback.format_exc()
        print(name, 'FAILED', flush=True)
        print(text, flush=True)
        os.makedirs(f'{WORK}/{name}', exist_ok=True)
        open(f'{WORK}/{name}/error.txt', 'w').write(text)


def is_done(name):
    return os.path.exists(f'{WORK}/{name}/summary.json')


def last_line(path):
    if not os.path.exists(path):
        return 'no output yet'
    with open(path, errors='replace') as f:
        text = f.read()[-3000:]
    lines = [l.strip() for l in re.split(r'[\r\n]+', text) if l.strip()]
    return lines[-1][:200] if lines else 'no output yet'


pending = [n for n in RUN_ORDER if not is_done(n) and not (n in LIVE and LIVE[n].is_alive())]
for n in RUN_ORDER:
    if n in LIVE and LIVE[n].is_alive():
        print(n, 'still running from before, waiting for it', flush=True)

while True:
    alive = [n for n in LIVE if LIVE[n].is_alive()]
    busy = {GPU_OF[n] for n in alive}
    for g in GPUS:
        if g not in busy and pending:
            n = pending.pop(0)
            if os.path.exists(f'{WORK}/{n}/error.txt'):
                os.remove(f'{WORK}/{n}/error.txt')
            t = threading.Thread(target=run_safe, args=(n, g), name=n)
            t.start()
            LIVE[n] = t
            GPU_OF[n] = g
            busy.add(g)
            print('started', n, 'on gpu', g, flush=True)
    alive = [n for n in LIVE if LIVE[n].is_alive()]
    if not alive and not pending:
        break
    clear_output(wait=True)
    for n in alive:
        print(n, 'gpu', GPU_OF[n], ':', last_line(f'{WORK}/{n}/train.log'), flush=True)
    if pending:
        print('waiting for a free gpu:', pending, flush=True)
    time.sleep(PROGRESS_EVERY)

errors = {}
for n in RUN_ORDER:
    if is_done(n):
        print(n, 'finished')
    else:
        err = f'{WORK}/{n}/error.txt'
        errors[n] = open(err).read() if os.path.exists(err) else 'no error file. the thread stopped early'
        print(n, 'FAILED, see the error file in', WORK)
json.dump(errors, open(f'{WORK}/train_errors.json', 'w'), indent=2)
print('done:', sorted(n for n in RUN_ORDER if is_done(n)))""")

md("## Merge LoRA and OPLoRA into full models\nThe merge runs on the CPU, so it does not need a GPU. The merged fp16 models are saved, so local evaluation does not need the adapter code. The merge is checked against the unmerged model on a few texts.")

code(r"""def build_merged(name):
    path = f'{WORK}/merged/{name}'
    check_file = f'{path}/merge_check.json'
    if os.path.exists(check_file):
        return json.load(open(check_file))
    data = torch.load(f'{WORK}/{name}/snapshot_{UPDATES}/adapters.pt', weights_only=False)
    texts = ['Q: Tom has 3 apples and buys 4 more. How many apples does he have now?\nA:',
             'The capital of France is', 'def add(a, b):\n    return', 'Once upon a time, a small fox']
    enc = tok(texts, return_tensors='pt', padding=True)

    ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ref_mods = dict(ref.named_modules())
    handles = []
    for mname, m in data['modules'].items():
        A = m['A_eff'].float()
        B = m['B_eff'].float()
        scale = m['scale']
        def hook(mod, inputs, output, A=A, B=B, scale=scale):
            x = inputs[0].float()
            return output + (scale * ((x @ A.T) @ B.T)).to(output.dtype)
        handles.append(ref_mods[mname].register_forward_hook(hook))
    with torch.no_grad():
        ref_logits = ref(**enc).logits.float()
    for h in handles:
        h.remove()

    merged = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    mods = dict(merged.named_modules())
    with torch.no_grad():
        for mname, m in data['modules'].items():
            A = m['A_eff'].float()
            B = m['B_eff'].float()
            mods[mname].weight.add_(m['scale'] * (B @ A))
        merged_logits = merged(**enc).logits.float()
    diff = float((merged_logits - ref_logits).abs().max())
    report = {'run': name, 'max_abs_logit_diff_fp32': diff,
              'relative_diff_fp32': diff / float(ref_logits.abs().max()),
              'modules_merged': len(data['modules'])}

    merged.half()
    os.makedirs(path, exist_ok=True)
    merged.save_pretrained(path)
    tok.save_pretrained(path)
    json.dump(report, open(check_file, 'w'), indent=2)
    del ref, merged, data
    gc.collect()
    return report

merge_reports = {n: build_merged(n) for n in RUN_ORDER if os.path.exists(f'{WORK}/{n}/summary.json')}
print(json.dumps(merge_reports, indent=2))""")

md("## Training curves\nSFT loss and learning rate against the update, for every finished run.")

code(r"""fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name in RUN_ORDER:
    path = f'{WORK}/{name}/train_log.json'
    if os.path.exists(path):
        log = json.load(open(path))
        ax[0].plot([r['update'] for r in log], [r['sft_loss'] for r in log], label=name)
        ax[1].plot([r['update'] for r in log], [r['lr'] for r in log], label=name)
ax[0].set_title('sft loss per update')
ax[0].set_xlabel('update')
ax[0].set_ylabel('sft loss per target token')
ax[1].set_title('learning rate per update')
ax[1].set_xlabel('update')
ax[1].set_ylabel('learning rate')
for a in ax:
    a.legend()
fig.tight_layout()
os.makedirs(f'{WORK}/report', exist_ok=True)
fig.savefig(f'{WORK}/report/fig_training_curves_lora.png', dpi=120)
plt.close(fig)
print('saved fig_training_curves_lora.png')""")

md("## Config, manifest and zip\nThe zip holds everything under `/kaggle/working/task1`, including the merged models and the recovery files. Download the zip from the Output tab. It is then evaluated on a local machine.")

code(r"""config = {
    'model': MODEL, 'seed': SEED, 'num_examples': NUM_EXAMPLES, 'effective_batch': EFFECTIVE_BATCH,
    'micro_batch': MICRO_BATCH, 'max_len': MAX_LEN, 'updates': UPDATES, 'warmup': WARMUP,
    'optimizer': {'name': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.01,
                  'decay_rule': 'tensors with 2 or more dims'},
    'schedule': 'cosine with linear warmup', 'grad_clip_norm': 1.0,
    'grad_scaler_init_scale': 2 ** 11,
    'precision': 'frozen fp16 backbone, fp32 adapters and optimizer state, fp16 autocast on cuda',
    'lora': {'dropout_on_input': DROPOUT, 'targets': TARGETS,
             'a_init': 'uniform in +-1/sqrt(in_features)', 'b_init': 'zeros'},
    'protect_k': PROTECT_K, 'recovery_steps': sorted(RECOVERY_STEPS), 'run_order': RUN_ORDER,
    'runs': {n: {'method': m, 'rank': r, 'alpha': a, 'peak_lr': lr} for n, (m, r, a, lr) in RUNS.items()},
    'gpus': GPUS, 'cuda': USE_CUDA,
    'versions': {m: pkg.version(m) for m in ['torch', 'transformers', 'datasets', 'numpy']},
    'python': sys.version.split()[0],
}
json.dump(config, open(f'{WORK}/config.json', 'w'), indent=2)

readme = '''task 1 lora and oplora training outputs

<run>/snapshot_1250/adapters.pt     raw and effective LoRA factors. oplora also keeps its bases and the leakage checks
<run>/snapshot_1250/leakage.json    oplora leakage checks (empty for lora)
<run>/snapshot_1250/ tokenizer files
<run>/train_log.json                per update log: sft loss, lr, grad norm, skipped, seconds, peak memory
<run>/summary.json                  run summary: params, times, memory, file sizes, checks
<run>/recovery/recovery.pt          state at the last save point (updates 312, 625, 937, 1250)
merged/<run>/                       fp16 merged models for evaluation, with merge_check.json
report/fig_training_curves_lora.png sft loss and learning rate
train_examples.json                 source row numbers of the 40,000 training examples
config.json                         settings and versions
manifest.json                       every file with its size
train_errors.json                   failures, if any
'''
open(f'{WORK}/README.txt', 'w').write(readme)

files = []
for root, _, names in os.walk(WORK):
    for n in names:
        p = os.path.join(root, n)
        files.append({'path': os.path.relpath(p, WORK), 'bytes': os.path.getsize(p)})
json.dump(sorted(files, key=lambda f: f['path']), open(f'{WORK}/manifest.json', 'w'), indent=1)

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, names in os.walk(WORK):
        for n in names:
            p = os.path.join(root, n)
            z.write(p, os.path.relpath(p, '/kaggle/working'))
folder_gb = dir_bytes(WORK) / 1e9
zip_gb = os.path.getsize(ZIP_PATH) / 1e9
print(f'zip: {ZIP_PATH} {zip_gb:.2f} GB | folder: {folder_gb:.2f} GB | output total: {folder_gb + zip_gb:.2f} GB of 20 GB')""")

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
