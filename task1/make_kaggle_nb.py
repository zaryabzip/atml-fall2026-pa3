import json

CELLS = []


def md(text):
    CELLS.append(("md", text))


def code(text):
    CELLS.append(("code", text))


md("""# PA3 task 1: full FT, L2-SP, LoRA-4, LoRA-16, OPLoRA-16

Everything runs in this notebook: data, training, merging, evaluation, diagnostics, qualitative outputs and the report.
Settings: Internet on, accelerator GPU T4 x2.

Every step saves its outputs under `/kaggle/working/task1`. A step that already has its outputs is skipped when you rerun the notebook.

Training resumes from its last recovery point (updates 312, 625 and 937) if the cell is interrupted or the kernel restarts. The recovery files of Full FT and L2-SP are about 7 GB each, so they go to `/kaggle/temp`, which is not part of the 20 GB output. A full session restart can lose `/kaggle/temp`, so resume across sessions is not guaranteed for those two runs.
`/kaggle/working/task1/README.txt` lists every saved file. Do not stop a training cell while it runs, because stopping it kills the runs.""")

md("## Setup")

code(r"""import os, gc, sys, json, time, random, re, math, shutil, threading, traceback, zipfile, importlib.metadata as pkg
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt

MODEL = 'Qwen/Qwen3-0.6B'
WORK = '/kaggle/working/task1'
SEED = 6304
NUM_EXAMPLES = 40000
EFFECTIVE_BATCH = 32
MICRO_BATCH = 2
MAX_LEN = 2048
UPDATES = 1250
WARMUP = 38
CE_CHUNK = 512
LAM = 1e-3
PROTECT_K = 128
ANCHOR_LAYERS = (0, 13, 27)
RUNS = {  # name: (method, rank, alpha, peak lr)
    'full': ('full', 0, 0, 2e-5),
    'l2sp': ('l2sp', 0, 0, 2e-5),
    'lora4': ('lora', 4, 8, 2e-4),
    'lora16': ('lora', 16, 32, 2e-4),
    'oplora16': ('oplora', 16, 32, 2e-4),
}
SNAP_STEPS = {'full': {UPDATES // 2, UPDATES}, 'l2sp': {UPDATES}, 'lora4': {UPDATES},
              'lora16': {UPDATES}, 'oplora16': {UPDATES}}
RECOVERY_STEPS = {312, 625, 937}  # save points from the spec. each run keeps one recovery file that is overwritten
CRASH_AT = {}  # test hook only, keep empty
PARALLEL = True  # two runs at once, one per gpu. False runs them one after another.
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LOAD_LOCK = threading.Lock()  # loading models in two threads at once failed in testing
CKPT_LOCK = threading.Lock()  # one recovery write at a time, across both runs

for d in [WORK, f'{WORK}/merged', f'{WORK}/eval', f'{WORK}/diagnostics', f'{WORK}/report', f'{WORK}/qualitative']:
    os.makedirs(d, exist_ok=True)
for p in [WORK, '/kaggle/temp']:
    os.makedirs(p, exist_ok=True)
    u = shutil.disk_usage(p)
    print(f'{p}: free {u.free / 1e9:.1f} GB of {u.total / 1e9:.1f} GB')
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
print('device:', DEVICE, '| gpus:', torch.cuda.device_count() if torch.cuda.is_available() else 0)""")

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


def displacement(params, anchor, device):
    sq = 0.0
    with torch.no_grad():
        for p, a in zip(params, anchor):
            sq += float(((p.detach() - a.to(device)) ** 2).sum())
    return sq ** 0.5


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)


def save_snapshot(model, path):
    # full ft and l2sp: fp16 weights for eval, and fp32 down_proj copies for the diagnostics
    os.makedirs(path, exist_ok=True)
    sd = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
    if model.config.tie_word_embeddings:
        sd.pop('lm_head.weight', None)  # tied to the embedding, so saved once
    model.save_pretrained(path, state_dict=sd)
    tok.save_pretrained(path)
    torch.save({L: model.model.layers[L].mlp.down_proj.weight.detach().float().cpu() for L in ANCHOR_LAYERS},
               f'{path}/down_proj_fp32.pt')


class LoRALinear(nn.Module):
    # frozen linear plus a low rank update. with protect_k > 0 the update is projected (oplora).
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
        # leading singular bases of the starting weight, computed in fp32
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
    setup_seconds = 0.0
    idx = 0
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
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
    # raw and effective factors for every target, plus the oplora bases and leakage checks
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


def save_adapters(model, path, meta):
    os.makedirs(path, exist_ok=True)
    export_adapters(model, f'{path}/adapters.pt', meta)
    tok.save_pretrained(path)""")

md("## Training function\nOne function for all five runs. Full FT and L2-SP train the full model in fp32 with fp16 autocast. LoRA and OPLoRA freeze an fp16 backbone and train fp32 adapters. L2-SP adds lambda times (theta minus theta0) to the gradient after unscaling and before clipping.")

code(r"""def recovery_dir(name):
    if RUNS[name][0] in ('full', 'l2sp'):
        return f'/kaggle/temp/task1/{name}'  # large files, kept out of the working folder
    return f'{WORK}/{name}/recovery'


def save_recovery(name, step, named, opt, sched, scaler, device, extra):
    rdir = recovery_dir(name)
    os.makedirs(rdir, exist_ok=True)
    need = 1.1 * 3 * sum(p.numel() * p.element_size() for _, p in named)  # params, adam m and v
    free = shutil.disk_usage(rdir).free
    if free < need:
        raise RuntimeError(f'{name}: not enough disk for recovery. need {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB')
    state = {'step': step, 'trainable': {n: p.detach().cpu() for n, p in named},
             'opt': opt.state_dict(), 'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
             'rng_cpu': torch.get_rng_state(), 'extra': extra}
    if device.type == 'cuda':
        state['rng_cuda'] = torch.cuda.get_rng_state(device)
    with CKPT_LOCK:
        tmp = f'{rdir}/recovery.tmp'
        torch.save(state, tmp)
        os.replace(tmp, f'{rdir}/recovery.pt')
    del state


def load_recovery(name, model, opt, sched, scaler, device):
    path = f'{recovery_dir(name)}/recovery.pt'
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    lookup = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in ck['trainable'].items():
            lookup[n].copy_(v.to(device))
    opt.load_state_dict(ck['opt'])
    sched.load_state_dict(ck['sched'])
    scaler.load_state_dict(ck['scaler'])
    torch.set_rng_state(ck['rng_cpu'])
    if device.type == 'cuda' and 'rng_cuda' in ck:
        torch.cuda.set_rng_state(ck['rng_cuda'], device)
    return ck


def train_run(name, gpu):
    method, rank, alpha, lr = RUNS[name]
    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{gpu}') if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.cuda.set_device(device)
    out = f'{WORK}/{name}'
    os.makedirs(out, exist_ok=True)
    full_ft = method in ('full', 'l2sp')
    dtype = torch.float32 if (full_ft or not use_cuda) else torch.float16

    with LOAD_LOCK:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model = model.to(device)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    setup_seconds = 0.0
    if full_ft:
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        setup_seconds = apply_lora(model, rank, alpha, 0.05, PROTECT_K if method == 'oplora' else 0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    n_trainable = sum(p.numel() for p in params)
    anchor = [p.detach().float().cpu().clone() for p in params] if full_ft else None
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=use_cuda)
    sched = get_cosine_schedule_with_warmup(opt, WARMUP, UPDATES)
    scaler = torch.amp.GradScaler(device.type, enabled=use_cuda, init_scale=2 ** 11)
    amp = use_cuda

    log, snap_seconds, penalty_total, disp_at, start = [], {}, 0.0, {}, 0
    ck = load_recovery(name, model, opt, sched, scaler, device)
    if ck is not None:
        extra = ck['extra']
        log, snap_seconds, penalty_total = extra['log'], extra['snap_seconds'], extra['penalty_total']
        disp_at, setup_seconds, start = extra['disp_at'], extra['setup_seconds'], ck['step']
        print(name, 'resumed at update', start, flush=True)
        del ck
    bar = tqdm(total=UPDATES, initial=start, desc=name, position=0 if gpu == 0 else 1, leave=True)
    for step in range(start, UPDATES):
        t0 = time.time()
        batch = examples[step * EFFECTIVE_BATCH:(step + 1) * EFFECTIVE_BATCH]
        total_tokens = sum(len(a) for _, a in batch)  # loss is normalized over all response tokens
        sft_sum = torch.zeros((), device=device)
        for i in range(0, EFFECTIVE_BATCH, MICRO_BATCH):
            ids, att, lab = collate(batch[i:i + MICRO_BATCH], device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
                mask = lab[:, 1:] != -100
                mb_loss = chunked_ce(model.lm_head, hidden[:, :-1][mask], lab[:, 1:][mask])
            scaler.scale(mb_loss / total_tokens).backward()
            sft_sum += mb_loss.detach()

        scale_before = scaler.get_scale()
        scaler.unscale_(opt)
        omega, penalty_s = 0.0, 0.0
        if anchor is not None:
            tp = time.time()
            om = torch.zeros((), device=device)
            for p, a in zip(params, anchor):
                d = p.detach() - a.to(device)
                om += (d * d).sum()
                if method == 'l2sp':
                    p.grad.add_(d, alpha=LAM)
            omega = 0.5 * float(om)
            penalty_s = time.time() - tp
            penalty_total += penalty_s
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        lr_now = opt.param_groups[0]['lr']
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        opt.zero_grad(set_to_none=True)
        sched.step()

        sft = float(sft_sum) / total_tokens
        rec = {'update': step + 1, 'sft_loss': sft, 'omega': omega,
               'lam_omega': LAM * omega if method == 'l2sp' else 0.0,
               'lr': lr_now, 'grad_norm': float(grad_norm), 'skipped': skipped,
               'seconds': round(time.time() - t0, 2), 'penalty_seconds': round(penalty_s, 3)}
        if use_cuda:
            rec['peak_mem_GB'] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        log.append(rec)
        bar.update(1)
        bar.set_postfix(sft=f'{sft:.4f}', lr=f'{lr_now:.1e}', s=rec['seconds'])

        if step + 1 in SNAP_STEPS[name]:
            t_snap = time.time()
            if full_ft:
                save_snapshot(model, f'{out}/snapshot_{step + 1}')
            else:
                save_adapters(model, f'{out}/snapshot_{step + 1}',
                              {'method': method, 'rank': rank, 'alpha': alpha,
                               'protect_k': PROTECT_K if method == 'oplora' else 0})
            snap_seconds[f'snapshot_{step + 1}'] = round(time.time() - t_snap, 2)
            disp_at[str(step + 1)] = displacement(params, anchor, device) if anchor else None
        if step + 1 in RECOVERY_STEPS:
            extra = {'log': log, 'snap_seconds': snap_seconds, 'penalty_total': penalty_total,
                     'setup_seconds': setup_seconds, 'disp_at': disp_at}
            save_recovery(name, step + 1, named, opt, sched, scaler, device, extra)
        if CRASH_AT.get(name) == step + 1:
            raise RuntimeError(f'test crash at update {step + 1}')
    bar.close()

    summary = {'run': name, 'method': method, 'rank': rank, 'alpha': alpha, 'lr': lr,
               'trainable_params': n_trainable,
               'train_seconds': round(sum(r['seconds'] for r in log), 1),
               'peak_mem_GB': round(max(r.get('peak_mem_GB', 0) for r in log), 2) if use_cuda else None,
               'setup_seconds': round(setup_seconds, 2),
               'anchor_bytes': sum(a.numel() * a.element_size() for a in anchor) if anchor else 0,
               'basis_bytes': (sum((m.Uk.numel() + m.Vk.numel()) * 4 for m in lora_modules(model))
                               if method == 'oplora' else 0),
               'penalty_seconds_total': round(penalty_total, 2),
               'snapshot_seconds': snap_seconds,
               'displacement_at': disp_at,
               'final_displacement': displacement(params, anchor, device) if anchor else None,
               'snapshot_bytes': dir_bytes(out)}
    json.dump(log, open(f'{out}/train_log.json', 'w'))
    json.dump(summary, open(f'{out}/summary.json', 'w'), indent=2)
    shutil.rmtree(recovery_dir(name), ignore_errors=True)
    del model, opt, sched, scaler, params, anchor
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return log, summary""")

md("## Train\nRuns that already have a `summary.json` are skipped. Full FT and L2-SP run first, then LoRA-4 and LoRA-16, then OPLoRA-16. Leave the cell running until it prints the finish message.")

code(r"""results = globals().setdefault('results', {})  # kept across reruns of this cell
errors = globals().setdefault('errors', {})
LIVE = globals().setdefault('LIVE', {})  # training threads of this kernel

def run(name, gpu):
    try:
        results[name] = train_run(name, gpu)
    except Exception:
        errors[name] = traceback.format_exc()

def run_group(names):
    finished = [n for n in names if os.path.exists(f'{WORK}/{n}/summary.json')]
    todo = [n for n in names if n not in finished]
    if finished:
        print('finished, skipping:', finished)
    waiting = [n for n in todo if n in LIVE and LIVE[n].is_alive()]
    if waiting:
        print('still running from before, waiting for:', waiting)
    to_start = [n for n in todo if n not in waiting]
    if PARALLEL:
        for i, n in enumerate(to_start):
            errors.pop(n, None)
            LIVE[n] = threading.Thread(target=run, args=(n, i % 2), name=n)
            LIVE[n].start()
        for n in todo:
            LIVE[n].join()
    else:
        for n in to_start:
            errors.pop(n, None)
            run(n, 0)
        for n in waiting:
            LIVE[n].join()

run_group(['full', 'l2sp'])
run_group(['lora4', 'lora16'])
run_group(['oplora16'])

for name, msg in errors.items():
    print(name, 'FAILED:')
    print(msg)
print('finished:', sorted(n for n in RUNS if os.path.exists(f'{WORK}/{n}/summary.json')))""")

md("## Training curves")

code(r"""def load_log(name):
    return json.load(open(f'{WORK}/{name}/train_log.json'))

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for name in RUNS:
    if os.path.exists(f'{WORK}/{name}/train_log.json'):
        log = load_log(name)
        ax[0].plot([r['update'] for r in log], [r['sft_loss'] for r in log], label=name)
        if name in ('full', 'l2sp'):
            ax[1].plot([r['update'] for r in log], [r['omega'] for r in log], label=name)
            ax[2].plot([r['update'] for r in log], [r['lam_omega'] for r in log], label=name)
ax[0].set_title('sft loss, five runs')
ax[1].set_title('omega, half squared distance from start')
ax[2].set_title('lambda times omega (l2sp)')
for a in ax:
    a.set_xlabel('update')
    a.legend()
plt.tight_layout()
plt.savefig(f'{WORK}/report/fig_training_curves.png', dpi=120)
plt.show()""")

md("## Merge LoRA and OPLoRA into full models\nThe merged fp16 models are saved, so the evaluation can run later without this step. The merge is checked against the unmerged model on a few texts.")

code(r"""def build_merged(name):
    path = f'{WORK}/merged/{name}'
    check_file = f'{path}/merge_check.json'
    if os.path.exists(check_file):
        return json.load(open(check_file))
    data = torch.load(f'{WORK}/{name}/snapshot_{UPDATES}/adapters.pt', weights_only=False)
    device = DEVICE
    texts = ['Q: Tom has 3 apples and buys 4 more. How many apples does he have now?\nA:',
             'The capital of France is', 'def add(a, b):\n    return', 'Once upon a time, a small fox']
    enc = tok(texts, return_tensors='pt', padding=True).to(device)

    with LOAD_LOCK:
        ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    ref = ref.to(device).eval()
    ref_mods = dict(ref.named_modules())
    handles = []
    for mname, m in data['modules'].items():
        A = m['A_eff'].to(device, torch.float32)
        B = m['B_eff'].to(device, torch.float32)
        scale = m['scale']
        def hook(mod, inputs, output, A=A, B=B, scale=scale):
            x = inputs[0].float()
            return output + (scale * ((x @ A.T) @ B.T)).to(output.dtype)
        handles.append(ref_mods[mname].register_forward_hook(hook))
    with torch.no_grad():
        ref_logits = ref(**enc).logits.float().cpu()
    for h in handles:
        h.remove()

    with LOAD_LOCK:
        merged = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    merged = merged.to(device).eval()
    mods = dict(merged.named_modules())
    with torch.no_grad():
        for mname, m in data['modules'].items():
            A = m['A_eff'].to(device, torch.float32)
            B = m['B_eff'].to(device, torch.float32)
            mods[mname].weight.add_(m['scale'] * (B @ A))
        merged_logits = merged(**enc).logits.float().cpu()
    diff = float((merged_logits - ref_logits).abs().max())
    report = {'run': name, 'max_abs_logit_diff_fp32': diff,
              'relative_diff_fp32': diff / float(ref_logits.abs().max()),
              'modules_merged': len(data['modules'])}

    merged.half()
    os.makedirs(path, exist_ok=True)
    merged.save_pretrained(path)
    tok.save_pretrained(path)
    json.dump(report, open(check_file, 'w'), indent=2)
    del ref, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report

merge_reports = {n: build_merged(n) for n in ['lora4', 'lora16', 'oplora16'] if os.path.exists(f'{WORK}/{n}/summary.json')}
print(json.dumps(merge_reports, indent=2))""")

md("## Evaluation setup\nBenchmarks: GSM8K (chat template, non thinking, 0 shot, greedy, 1024 tokens), HellaSwag, BoolQ and ARC-Easy (no chat template). Per-example outputs are saved for every benchmark. Finished benchmarks are skipped.")

code("%pip install -q lm_eval==0.4.13")

code(r"""import lm_eval

MAX_NEW = 1024
EVAL_LIMIT = None  # None means the full benchmark
BENCH = [  # name, tasks, chat template, primary metric key
    ('gsm8k', ['gsm8k_cot'], True, 'exact_match,flexible-extract'),
    ('hellaswag', ['hellaswag'], False, 'acc_norm,none'),
    ('boolq', ['boolq'], False, 'acc,none'),
    ('arc_easy', ['arc_easy'], False, 'acc_norm,none'),
]
EVAL_VARIANTS = {
    'baseline': MODEL,
    'full_mid': f'{WORK}/full/snapshot_{UPDATES // 2}',
    'full_final': f'{WORK}/full/snapshot_{UPDATES}',
    'l2sp_final': f'{WORK}/l2sp/snapshot_{UPDATES}',
    'lora4_final': f'{WORK}/merged/lora4',
    'lora16_final': f'{WORK}/merged/lora16',
    'oplora16_final': f'{WORK}/merged/oplora16',
}
EVAL_FILE = f'{WORK}/eval/eval_results.json'
os.makedirs(f'{WORK}/eval/cache', exist_ok=True)
eval_out = json.load(open(EVAL_FILE)) if os.path.exists(EVAL_FILE) else {}
json.dump({'max_gen_toks': MAX_NEW, 'limit': EVAL_LIMIT, 'lm_eval': pkg.version('lm_eval'),
           'benchmarks': [b[0] for b in BENCH], 'variants': EVAL_VARIANTS},
          open(f'{WORK}/eval/eval_config.json', 'w'), indent=2)
device_str = str(DEVICE)

for vname, path in EVAL_VARIANTS.items():
    if path.startswith('/') and not os.path.exists(path):
        print('missing model, skipping', vname, path)
        continue
    eval_out.setdefault(vname, {})
    for bname, tasks, chat, key in BENCH:
        if bname in eval_out[vname]:
            continue
        model_args = f'pretrained={path},dtype=float16' + (',enable_thinking=False' if chat else '')
        res = lm_eval.simple_evaluate(
            model='hf', model_args=model_args, tasks=tasks, num_fewshot=0,
            batch_size=32 if chat else 8, device=device_str, apply_chat_template=chat,
            gen_kwargs={'max_gen_toks': MAX_NEW, 'do_sample': False} if chat else None,
            log_samples=True, limit=EVAL_LIMIT, use_cache=f'{WORK}/eval/cache/{vname}_{bname}')
        metrics = res['results'][tasks[0]]
        eval_out[vname][bname] = metrics
        json.dump(eval_out, open(EVAL_FILE, 'w'), indent=2, default=str)
        print(vname, bname, round(metrics[key], 4))

        samples = res.get('samples', {}).get(tasks[0], [])
        with open(f'{WORK}/eval/{vname}_{bname}_samples.jsonl', 'w') as f:
            for s in samples:
                response = s['resps'][0][0] if s.get('resps') else ''
                row = {'doc_id': s.get('doc_id'), 'filter': s.get('filter'), 'target': s.get('target'),
                       'response': response, 'filtered': s.get('filtered_resps')}
                if chat:
                    row['truncated'] = len(tok(response, add_special_tokens=False).input_ids) >= MAX_NEW - 16
                f.write(json.dumps(row, default=str) + '\n')
print('evaluation done for:', sorted(eval_out))""")

md("## Diagnostics\nWeight space: rho_k and eta for layers 0, 13 and 27 (down_proj). Activations: R(z) on a seeded probe set of 32 GSM8K test questions and 32 HellaSwag validation contexts. The baseline activations and the probe IDs are saved, so this part can be rerun without the baseline model.")

code(r"""import subprocess
LAYERS = list(ANCHOR_LAYERS)
rng = np.random.default_rng(SEED)
hs = load_dataset('Rowan/hellaswag', split='validation')
gi = sorted(int(i) for i in rng.choice(len(gsm8k), 32, replace=False))
hi = sorted(int(i) for i in rng.choice(len(hs), 32, replace=False))
json.dump({'gsm8k_test_rows': gi, 'hellaswag_validation_rows': hi},
          open(f'{WORK}/diagnostics/probe_ids.json', 'w'))

def hs_preprocess(text):
    text = text.strip().replace(' [title]', '. ')
    text = re.sub(r'\[.*?\]', '', text)
    return text.replace('  ', ' ')

def hs_query(doc):
    ctx = doc['ctx_a'] + ' ' + doc['ctx_b'].capitalize()
    return hs_preprocess(doc['activity_label'] + ': ' + ctx)

prompts = {
    'math': [tok.apply_chat_template([{'role': 'user', 'content': 'Q: ' + gsm8k[i]['question'] + '\nA:'}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in gi],
    'non_math': [hs_query(hs[i]) for i in hi],
}

act_file = f'{WORK}/diagnostics/activations_baseline.pt'
if os.path.exists(act_file):
    saved = torch.load(act_file, weights_only=False)
    z, W0 = saved['z'], saved['W0']
else:
    with LOAD_LOCK:
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    base = base.to(DEVICE).eval()
    W0 = {L: base.model.layers[L].mlp.down_proj.weight.detach().float().cpu() for L in LAYERS}
    z = {dom: {L: [] for L in LAYERS} for dom in prompts}
    for dom, plist in prompts.items():
        for p in plist:
            enc = tok(p, return_tensors='pt', add_special_tokens=False).to(DEVICE)
            store = {}
            hooks = [base.model.layers[L].mlp.down_proj.register_forward_pre_hook(
                         lambda m, inp, L=L: store.__setitem__(L, inp[0].detach())) for L in LAYERS]
            with torch.no_grad():
                base(**enc)
            for h in hooks:
                h.remove()
            last = enc.input_ids.shape[1] - 1  # last token of the prompt, before any answer
            for L in LAYERS:
                z[dom][L].append(store[L][0, last].float().cpu())
    z = {dom: {L: torch.stack(z[dom][L]) for L in LAYERS} for dom in z}
    torch.save({'z': z, 'W0': W0}, act_file)
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

bases = {L: torch.linalg.svd(W0[L].half().float(), full_matrices=False)[0][:, :PROTECT_K] for L in LAYERS}
DIAG_VARIANTS = {
    'full_mid': ('snap', f'{WORK}/full/snapshot_{UPDATES // 2}'),
    'full_final': ('snap', f'{WORK}/full/snapshot_{UPDATES}'),
    'l2sp_final': ('snap', f'{WORK}/l2sp/snapshot_{UPDATES}'),
    'lora4_final': ('adapt', f'{WORK}/lora4/snapshot_{UPDATES}/adapters.pt'),
    'lora16_final': ('adapt', f'{WORK}/lora16/snapshot_{UPDATES}/adapters.pt'),
    'oplora16_final': ('adapt', f'{WORK}/oplora16/snapshot_{UPDATES}/adapters.pt'),
}
weight_rows, act_rows = [], []
for vname, (kind, path) in DIAG_VARIANTS.items():
    if not os.path.exists(path if kind == 'adapt' else f'{path}/down_proj_fp32.pt'):
        print('missing, skipping', vname)
        continue
    if kind == 'snap':
        trained = torch.load(f'{path}/down_proj_fp32.pt', weights_only=False)
    else:
        adapters = torch.load(path, weights_only=False)
    for L in LAYERS:
        w0 = W0[L]
        if kind == 'snap':
            dw = trained[L].float() - w0
        else:
            m = adapters['modules'][f'model.layers.{L}.mlp.down_proj']
            dw = m['scale'] * (m['B_eff'].float() @ m['A_eff'].float())
        Uk = bases[L]
        rho = float((Uk.T @ dw).pow(2).sum() / dw.pow(2).sum().clamp(min=1e-30))
        eta = float(dw.norm() / w0.norm())
        weight_rows.append({'variant': vname, 'layer': L, 'rho_k': rho, 'eta': eta,
                            'delta_fro': float(dw.norm()), 'w0_fro': float(w0.norm())})
        for dom in ['math', 'non_math']:
            zz = z[dom][L]
            num = (zz @ dw.T).norm(dim=1)
            den = (zz @ w0.T).norm(dim=1) + 1e-8
            for i, r in enumerate((num / den).tolist()):
                act_rows.append({'variant': vname, 'layer': L, 'domain': dom, 'prompt': i, 'R': r})

wdf = pd.DataFrame(weight_rows)
adf = pd.DataFrame(act_rows)
wdf.to_csv(f'{WORK}/diagnostics/weight_space.csv', index=False)
adf.to_csv(f'{WORK}/diagnostics/activation_per_prompt.csv', index=False)
act_summary = adf.groupby(['variant', 'layer', 'domain'])['R'].agg(['mean', 'median']).reset_index()
act_summary.to_csv(f'{WORK}/diagnostics/activation_summary.csv', index=False)
print(wdf.round(6).to_string(index=False))
print(act_summary.round(6).to_string(index=False))""")

md("## Qualitative outputs\nNeeds `first_plot_questions.yaml` in `/kaggle/working/task1`. This is 24 prompts, 8 each in plain, JSON and template format. Without it this section is skipped, and the rest of the notebook still runs.")

code(r"""import csv
import yaml

QUAL_MAX_NEW = 512
QUAL_FILE = f'{WORK}/first_plot_questions.yaml'
QFMTS = ['plain', 'json', 'template']

def generate_variant(vname, path, qrows):
    with LOAD_LOCK:
        qm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32)
    qm = qm.to(DEVICE).eval()
    qtok = AutoTokenizer.from_pretrained(path)
    qtok.padding_side = 'left'
    out = []
    for i in range(0, len(qrows), 8):
        chunk = qrows[i:i + 8]
        texts = [qtok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False) for r in chunk]
        enc = qtok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(DEVICE)
        with torch.no_grad():
            gen = qm.generate(**enc, max_new_tokens=QUAL_MAX_NEW, do_sample=False, use_cache=True,
                              pad_token_id=qtok.pad_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        for r, ids in zip(chunk, new):
            n = int((ids != qtok.pad_token_id).sum())
            out.append({'variant': vname, **r, 'response': qtok.decode(ids, skip_special_tokens=True).strip(),
                        'truncated': n >= QUAL_MAX_NEW})
    del qm
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return out

if not os.path.exists(QUAL_FILE):
    print('first_plot_questions.yaml not found in', WORK, '. qualitative section skipped.')
else:
    data = yaml.safe_load(open(QUAL_FILE))
    items = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
    qrows = []
    for i, it in enumerate(items):
        text = next(it[k] for k in ['prompt', 'text', 'question'] if k in it)
        fmt = next(str(it[k]).lower() for k in ['format', 'type', 'style'] if k in it)
        qid = next((it[k] for k in ['id', 'question_id', 'qid'] if k in it), i // 3)
        qrows.append({'question_id': str(qid), 'format': fmt, 'prompt': text})
    assert len(qrows) == 24, f'expected 24 prompts, got {len(qrows)}'
    for f_ in QFMTS:
        assert len({r['question_id'] for r in qrows if r['format'] == f_}) == 8, f'format {f_} needs 8 ids'
    qorder = list(dict.fromkeys(r['question_id'] for r in qrows))
    records = []
    for vname, path in EVAL_VARIANTS.items():
        if path.startswith('/') and not os.path.exists(path):
            print('missing model, skipping', vname)
            continue
        vfile = f'{WORK}/qualitative/{vname}.json'
        if os.path.exists(vfile):
            records += json.load(open(vfile))
            continue
        recs = generate_variant(vname, path, qrows)
        json.dump(recs, open(vfile, 'w'), indent=2)
        records += recs
    vorder = list(EVAL_VARIANTS)
    records.sort(key=lambda r: (qorder.index(r['question_id']), QFMTS.index(r['format']), vorder.index(r['variant'])))
    json.dump(records, open(f'{WORK}/qualitative/responses.json', 'w'), indent=2)

    plain_ids = [r['question_id'] for r in qrows if r['format'] == 'plain']
    keep = {plain_ids[0], plain_ids[-1]}  # first and last plain questions
    fields = ['variant', 'question_id', 'format', 'prompt', 'response', 'truncated', 'format_compliance',
              'coherence', 'repetition', 'unnecessary_math', 'inappropriate', 'excerpt']
    with open(f'{WORK}/qualitative/annotation_subset.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            if r['question_id'] in keep:
                w.writerow({k: r.get(k, '') for k in fields})
    lines = ['# aligned excerpts', '', 'same question and format across variants, first 300 characters', '']
    for qid in [plain_ids[0], plain_ids[-1]]:
        for f_ in QFMTS:
            block = [r for r in records if r['question_id'] == qid and r['format'] == f_]
            if block:
                lines += [f'## question {qid}, {f_}', '', 'prompt: ' + block[0]['prompt'], '']
                for r in block:
                    flag = ' (truncated)' if r['truncated'] else ''
                    lines.append(f"- {r['variant']}{flag}: " + r['response'][:300].replace(chr(10), ' '))
                lines.append('')
    open(f'{WORK}/qualitative/excerpts.md', 'w').write(chr(10).join(lines))
    print('responses:', len(records), '| annotation rows:', sum(1 for r in records if r['question_id'] in keep))""")

md("## Report")

code(r"""def fmt(v, d=4):
    return '' if v is None else f'{v:.{d}f}'

PRIMARY = [('gsm8k', 'exact_match,flexible-extract', 'gsm8k flexible EM'),
           ('hellaswag', 'acc_norm,none', 'hellaswag acc_norm'),
           ('boolq', 'acc,none', 'boolq acc'),
           ('arc_easy', 'acc_norm,none', 'arc easy acc_norm')]
VARIANTS = ['baseline', 'full_mid', 'full_final', 'l2sp_final', 'lora4_final', 'lora16_final', 'oplora16_final']
COMPARE = [('lora4_final', 'lora16_final', 'rank 16 minus rank 4'),
           ('full_final', 'l2sp_final', 'l2sp minus full ft'),
           ('lora16_final', 'oplora16_final', 'oplora minus lora 16'),
           ('full_final', 'lora16_final', 'lora 16 minus full ft')]

def metric(vname, bname, key):
    try:
        return eval_out[vname][bname][key]
    except KeyError:
        return None

rows = []
for v in VARIANTS:
    rows.append([v] + [fmt(metric(v, b, k)) for b, k, _ in PRIMARY])
bench_table = pd.DataFrame(rows, columns=['variant'] + [lab for _, _, lab in PRIMARY])
delta_rows = []
for v in VARIANTS[1:]:
    delta_rows.append([v + ' delta pp'] + [
        '' if metric(v, b, k) is None or metric('baseline', b, k) is None
        else f'{(metric(v, b, k) - metric("baseline", b, k)) * 100:+.2f}' for b, k, _ in PRIMARY])
bench_table = pd.concat([bench_table, pd.DataFrame(delta_rows, columns=bench_table.columns)], ignore_index=True)
bench_table.to_csv(f'{WORK}/report/table_benchmark.csv', index=False)
print(bench_table.to_string(index=False))

cmp_rows = []
for a, b, label in COMPARE:
    cmp_rows.append([label] + [
        '' if metric(a, bn, k) is None or metric(b, bn, k) is None
        else f'{(metric(b, bn, k) - metric(a, bn, k)) * 100:+.2f}' for bn, k, _ in PRIMARY])
cmp_table = pd.DataFrame(cmp_rows, columns=['comparison'] + [lab + ' pp' for _, _, lab in PRIMARY])
cmp_table.to_csv(f'{WORK}/report/table_comparisons.csv', index=False)
print(cmp_table.to_string(index=False))

res_rows = []
for name in RUNS:
    if os.path.exists(f'{WORK}/{name}/summary.json'):
        s = json.load(open(f'{WORK}/{name}/summary.json'))
        res_rows.append({'run': name, 'trainable_params': s['trainable_params'],
                         'train_seconds': s['train_seconds'], 'peak_mem_GB': s['peak_mem_GB'],
                         'setup_seconds': s['setup_seconds'], 'basis_bytes': s['basis_bytes'], 'anchor_bytes': s['anchor_bytes'],
                         'penalty_seconds_total': s['penalty_seconds_total'],
                         'snapshot_seconds': round(sum(s['snapshot_seconds'].values()), 2),
                         'snapshot_bytes': s['snapshot_bytes']})
resources = pd.DataFrame(res_rows)
resources.to_csv(f'{WORK}/report/table_resources.csv', index=False)
print(resources.to_string(index=False))

penalty_rows = []
for name in ['full', 'l2sp']:
    if os.path.exists(f'{WORK}/{name}/summary.json'):
        s = json.load(open(f'{WORK}/{name}/summary.json'))
        last = load_log(name)[-1]
        penalty_rows.append({'run': name, 'final_displacement': s['final_displacement'],
                             'final_omega': last['omega'], 'final_lam_omega': last['lam_omega'],
                             'final_sft_loss': last['sft_loss']})
penalty_table = pd.DataFrame(penalty_rows)
penalty_table.to_csv(f'{WORK}/report/table_penalty_final.csv', index=False)
print(penalty_table.to_string(index=False))

merge_table = pd.DataFrame([{'run': k, **v} for k, v in merge_reports.items()])
merge_table.to_csv(f'{WORK}/report/table_merge_check.csv', index=False)
print(merge_table.to_string(index=False))

if os.path.exists(f'{WORK}/diagnostics/weight_space.csv'):
    diag_table = pd.read_csv(f'{WORK}/diagnostics/weight_space.csv').merge(
        pd.read_csv(f'{WORK}/diagnostics/activation_summary.csv').pivot_table(
            index=['variant', 'layer'], columns='domain', values='mean').reset_index(), on=['variant', 'layer'])
    diag_table.to_csv(f'{WORK}/report/table_diagnostics.csv', index=False)
    print(diag_table.round(6).to_string(index=False))

fig, ax = plt.subplots(1, len(PRIMARY), figsize=(16, 4))
for a, (bn, k, lab) in zip(ax, PRIMARY):
    vals = [metric(v, bn, k) or 0 for v in VARIANTS]
    a.bar(range(len(VARIANTS)), vals)
    a.set_xticks(range(len(VARIANTS)))
    a.set_xticklabels(VARIANTS, rotation=60, fontsize=7)
    a.set_title(lab)
plt.tight_layout()
plt.savefig(f'{WORK}/report/fig_benchmark.png', dpi=120)
plt.show()""")

md("## Extra tables\nRaw accuracy next to the normalized ones, midpoint displacement, and the OPLoRA leakage checks.")

code(r"""EXTRA_METRICS = [('gsm8k', 'exact_match,strict-match'), ('hellaswag', 'acc,none'),
                 ('boolq', 'acc,none'), ('arc_easy', 'acc,none')]
raw_table = pd.DataFrame([[v] + [fmt(metric(v, b, k)) for b, k in EXTRA_METRICS] for v in VARIANTS],
                         columns=['variant', 'gsm8k strict EM', 'hellaswag acc', 'boolq acc', 'arc easy acc'])
raw_table.to_csv(f'{WORK}/report/table_benchmark_raw.csv', index=False)
print(raw_table.to_string(index=False))

disp_rows = []
for name in ['full', 'l2sp']:
    if os.path.exists(f'{WORK}/{name}/summary.json'):
        s_ = json.load(open(f'{WORK}/{name}/summary.json'))
        disp_rows.append({'run': name, f'displacement_at_{UPDATES // 2}': s_['displacement_at'].get(str(UPDATES // 2)),
                          'final_displacement': s_['final_displacement']})
disp_table = pd.DataFrame(disp_rows)
disp_table.to_csv(f'{WORK}/report/table_displacement.csv', index=False)
print(disp_table.to_string(index=False))

leak_file = f'{WORK}/oplora16/snapshot_{UPDATES}/adapters.pt'
if os.path.exists(leak_file):
    leak_table = pd.DataFrame([{'run': 'oplora16', **torch.load(leak_file, weights_only=False)['checks']}])
    leak_table.to_csv(f'{WORK}/report/table_oplora_leakage.csv', index=False)
    print(leak_table.to_string(index=False))""")

md("## Manifest and zip\nThe zip holds everything, including the model weights, so the whole output can be tested locally. The zip is about as big as the folder, so the total is printed. Keep it under the 20 GB output limit.")

code(r"""files = []
for root, _, names in os.walk(WORK):
    for n in names:
        p = os.path.join(root, n)
        files.append({'path': os.path.relpath(p, WORK), 'bytes': os.path.getsize(p)})
config = {'model': MODEL, 'seed': SEED, 'num_examples': NUM_EXAMPLES, 'effective_batch': EFFECTIVE_BATCH,
          'micro_batch': MICRO_BATCH, 'max_len': MAX_LEN, 'updates': UPDATES, 'warmup': WARMUP,
          'lam': LAM, 'protect_k': PROTECT_K, 'runs': RUNS,
          'snap_steps': {k: sorted(v) for k, v in SNAP_STEPS.items()},
          'versions': {m: pkg.version(m) for m in ['torch', 'transformers', 'datasets', 'lm_eval', 'numpy', 'pandas']},
          'python': sys.version.split()[0]}
json.dump(config, open(f'{WORK}/config.json', 'w'), indent=2)
json.dump(files, open(f'{WORK}/manifest.json', 'w'), indent=1)
readme = '''task 1 outputs

<run>/snapshot_<step>/          full and l2sp: fp16 weights, down_proj_fp32.pt
<run>/snapshot_<step>/          lora and oplora: adapters.pt (raw and effective factors, oplora bases)
<run>/summary.json, train_log.json  training summary and per update log
merged/<run>/                   fp16 merged models for evaluation, with merge_check.json
eval/eval_results.json          benchmark metrics; delete an entry to rerun that benchmark
eval/eval_config.json           evaluation settings
eval/<variant>_<bench>_samples.jsonl  per example outputs
diagnostics/                    probe_ids.json, activations_baseline.pt, weight and activation csvs
report/                         tables and figures
train_examples.json             source row numbers of the 40,000 training examples
config.json, manifest.json      settings, versions and the file list
'''
open(f'{WORK}/README.txt', 'w').write(readme)

zip_path = '/kaggle/working/task1_outputs.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, names in os.walk(WORK):
        for n in names:
            p = os.path.join(root, n)
            z.write(p, os.path.relpath(p, '/kaggle/working'))
folder_gb = dir_bytes(WORK) / 1e9
zip_gb = os.path.getsize(zip_path) / 1e9
print(f'zip: {zip_path} {zip_gb:.2f} GB | folder: {folder_gb:.2f} GB | output total: {folder_gb + zip_gb:.2f} GB of 20 GB')
if folder_gb + zip_gb > 18:
    print('close to the 20 GB output limit. set DELETE_UNZIPPED_WEIGHTS = True below to delete the unzipped weights.')
DELETE_UNZIPPED_WEIGHTS = False  # true removes the unzipped weights after the zip is written
if DELETE_UNZIPPED_WEIGHTS:
    for root, _, names in os.walk(WORK):
        for n in names:
            if n.endswith('.safetensors'):
                os.remove(os.path.join(root, n))
    print('unzipped weights removed. the zip still holds them.')""")

COMPARE_ALL = """COMPARE = [('lora4_final', 'lora16_final', 'rank 16 minus rank 4'),
           ('full_final', 'l2sp_final', 'l2sp minus full ft'),
           ('lora16_final', 'oplora16_final', 'oplora minus lora 16'),
           ('full_final', 'lora16_final', 'lora 16 minus full ft')]"""
COMPARE_LORA = """COMPARE = [('lora4_final', 'lora16_final', 'rank 16 minus rank 4'),
           ('lora16_final', 'oplora16_final', 'oplora minus lora 16')]"""
COMPARE_FULL = "COMPARE = [('full_final', 'l2sp_final', 'l2sp minus full ft')]"
MERGE_REPORT_BLOCK = """merge_table = pd.DataFrame([{'run': k, **v} for k, v in merge_reports.items()])
merge_table.to_csv(f'{WORK}/report/table_merge_check.csv', index=False)
print(merge_table.to_string(index=False))
"""


MODES = {
    'lora': {
        'file': '/Users/zippy/Desktop/ATML/PA3/task1/kaggle_task1_lora.ipynb',
        'drop': [],
        'replace': [
            ('# PA3 task 1: full FT, L2-SP, LoRA-4, LoRA-16, OPLoRA-16', '# PA3 task 1: LoRA-4, LoRA-16, OPLoRA-16'),
            ("zip_path = '/kaggle/working/task1_outputs.zip'", "zip_path = '/kaggle/working/task1_lora_outputs.zip'"),
            ('One function for all five runs.', 'One function for all runs.'),
            ("ax[0].set_title('sft loss, five runs')", "ax[0].set_title('sft loss')"),
            ('The recovery files of Full FT and L2-SP are about 7 GB each, so they go to `/kaggle/temp`, which is not part of the 20 GB output. A full session restart can lose `/kaggle/temp`, so resume across sessions is not guaranteed for those two runs.',
             'LoRA recovery files are small and stay in the working folder.'),
            ("    'full': ('full', 0, 0, 2e-5),\n    'l2sp': ('l2sp', 0, 0, 2e-5),\n", ''),
            ('Full FT and L2-SP run first, then LoRA-4 and LoRA-16, then OPLoRA-16.', 'LoRA-4 and LoRA-16 run first, then OPLoRA-16.'),
            ("run_group(['full', 'l2sp'])\n", ''),
            ("    'full_mid': f'{WORK}/full/snapshot_{UPDATES // 2}',\n    'full_final': f'{WORK}/full/snapshot_{UPDATES}',\n    'l2sp_final': f'{WORK}/l2sp/snapshot_{UPDATES}',\n", ''),
            ("    'full_mid': ('snap', f'{WORK}/full/snapshot_{UPDATES // 2}'),\n    'full_final': ('snap', f'{WORK}/full/snapshot_{UPDATES}'),\n    'l2sp_final': ('snap', f'{WORK}/l2sp/snapshot_{UPDATES}'),\n", ''),
            ("VARIANTS = ['baseline', 'full_mid', 'full_final', 'l2sp_final', 'lora4_final', 'lora16_final', 'oplora16_final']",
             "VARIANTS = ['baseline', 'lora4_final', 'lora16_final', 'oplora16_final']"),
            (COMPARE_ALL, COMPARE_LORA),
        ],
    },
    'full': {
        'file': '/Users/zippy/Desktop/ATML/PA3/task1/kaggle_task1_full_l2sp.ipynb',
        'drop': ['## Merge LoRA and OPLoRA', 'def build_merged(name):'],
        'replace': [
            ('# PA3 task 1: full FT, L2-SP, LoRA-4, LoRA-16, OPLoRA-16', '# PA3 task 1: full FT and L2-SP'),
            ("zip_path = '/kaggle/working/task1_outputs.zip'", "zip_path = '/kaggle/working/task1_full_l2sp_outputs.zip'"),
            ('One function for all five runs.', 'One function for all runs.'),
            ("ax[0].set_title('sft loss, five runs')", "ax[0].set_title('sft loss')"),
            ("    'lora4': ('lora', 4, 8, 2e-4),\n    'lora16': ('lora', 16, 32, 2e-4),\n    'oplora16': ('oplora', 16, 32, 2e-4),\n", ''),
            ('Full FT and L2-SP run first, then LoRA-4 and LoRA-16, then OPLoRA-16.', 'Full FT and L2-SP run together, one per GPU.'),
            ("run_group(['lora4', 'lora16'])\nrun_group(['oplora16'])\n", ''),
            ("    'lora4_final': f'{WORK}/merged/lora4',\n    'lora16_final': f'{WORK}/merged/lora16',\n    'oplora16_final': f'{WORK}/merged/oplora16',\n", ''),
            ("    'lora4_final': ('adapt', f'{WORK}/lora4/snapshot_{UPDATES}/adapters.pt'),\n    'lora16_final': ('adapt', f'{WORK}/lora16/snapshot_{UPDATES}/adapters.pt'),\n    'oplora16_final': ('adapt', f'{WORK}/oplora16/snapshot_{UPDATES}/adapters.pt'),\n", ''),
            ("VARIANTS = ['baseline', 'full_mid', 'full_final', 'l2sp_final', 'lora4_final', 'lora16_final', 'oplora16_final']",
             "VARIANTS = ['baseline', 'full_mid', 'full_final', 'l2sp_final']"),
            (COMPARE_ALL, COMPARE_FULL),
            (MERGE_REPORT_BLOCK, ''),
        ],
    },
}


def build(mode):
    cfg = MODES[mode]
    kept = [(kind, text) for kind, text in CELLS if not any(s in text for s in cfg['drop'])]
    for old, new in cfg['replace']:
        hits = sum(text.count(old) for _, text in kept)
        assert hits == 1, f"{mode}: {hits} matches for {old[:70]!r}"
    nb_cells = []
    for kind, text in kept:
        for old, new in cfg['replace']:
            text = text.replace(old, new)
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
    with open(cfg['file'], "w") as f:
        json.dump(nb, f, indent=1)
    print(mode, "cells", len(nb_cells), "->", cfg['file'])


for mode in MODES:
    build(mode)
