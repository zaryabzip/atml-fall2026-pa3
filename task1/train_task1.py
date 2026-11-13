# task 1 training for full l2sp lora and oplora
# kaggle 2x t4 torchrun standalone nproc 2 method full
# single process run for local mps

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

# spec settings

SEED = 6304
EFFECTIVE_BATCH = 32
MICRO_BATCH = 2
MAX_LEN = 2048
WARMUP_UPDATES = 38
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAM_L2SP = 1e-3
PROTECT_K = 128


# command line

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["full", "l2sp", "lora", "oplora"])
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lam", type=float, default=LAM_L2SP)
    p.add_argument("--protect_k", type=int, default=PROTECT_K)
    p.add_argument("--num_examples", type=int, default=40_000)
    p.add_argument("--checkpoint_steps", default="312,625,937,1250")
    p.add_argument("--snapshot_steps", default=None)
    p.add_argument("--stop_at", type=int, default=None)
    p.add_argument("--recovery_dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--backbone_dtype", default="float16")
    return p.parse_args()


# data

def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def load_examples(tok, n):
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_questions = {normalize(q) for q in gsm8k["question"]}
    path = hf_hub_download("meta-math/MetaMathQA", "MetaMathQA-395K.json", repo_type="dataset")
    rows = json.load(open(path))
    kept = []
    for r in rows:
        query, response = r["query"], r["response"]
        if not query.strip() or not response.strip():
            continue
        if normalize(query) in gsm8k_questions or normalize(r.get("original_question", "")) in gsm8k_questions:
            continue
        prompt = tok.apply_chat_template([{"role": "user", "content": query}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(response + "<|im_end|>", add_special_tokens=False).input_ids
        if len(p_ids) + len(a_ids) > MAX_LEN:
            continue
        kept.append((p_ids, a_ids))
        if len(kept) == n:
            break
    if len(kept) < n:
        raise SystemExit(f"only {len(kept)} eligible examples, need {n}")
    return kept


# chunked cross entropy keeps only one chunk of vocab logits in memory
CE_CHUNK = 512


def _ce_part(head, h, t):
    return F.cross_entropy(head(h).float(), t, reduction="sum")


def chunked_ce(head, h, t):
    # logits are rebuilt per chunk during backward
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], CE_CHUNK):
        total = total + checkpoint(_ce_part, head, h[i:i + CE_CHUNK], t[i:i + CE_CHUNK], use_reentrant=False)
    return total


def collate(mb, pad_id, device):
    L = max(len(p) + len(a) for p, a in mb)
    ids = torch.full((len(mb), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(mb), L), dtype=torch.long)
    lab = torch.full((len(mb), L), -100, dtype=torch.long)
    for k, (p, a) in enumerate(mb):
        seq = p + a
        ids[k, : len(seq)] = torch.tensor(seq)
        att[k, : len(seq)] = 1
        lab[k, len(p): len(seq)] = torch.tensor(a)
    return ids.to(device), att.to(device), lab.to(device)


# lora and oplora linear layer

class LoRALinear(nn.Module):
    # frozen base plus low rank update. protect_k > 0 gives oplora.
    def __init__(self, base, rank, alpha, dropout, seed):
        super().__init__()
        self.base = base
        din, dout = base.in_features, base.out_features
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)
        g = torch.Generator().manual_seed(seed)
        # kaiming uniform with a equal to sqrt 5 gives bound one over sqrt of fan in
        A = (torch.rand(rank, din, generator=g) * 2 - 1) / math.sqrt(din)
        self.A = nn.Parameter(A.to(base.weight.device))
        self.B = nn.Parameter(torch.zeros(dout, rank, device=base.weight.device))
        self.protect_k = 0
        self.register_buffer("Uk", None, persistent=False)
        self.register_buffer("Vk", None, persistent=False)

    def set_bases(self, W0, k):
        # leading singular bases of the starting weight in fp32
        U, _, Vh = torch.linalg.svd(W0.float().cpu(), full_matrices=False)
        dev = self.A.device
        self.Uk = U[:, :k].contiguous().to(dev)
        self.Vk = Vh[:k].T.contiguous().to(dev)
        self.protect_k = k

    def effective(self):
        # projected factors for oplora raw for lora
        A, B = self.A, self.B
        if self.protect_k:
            A = A - (A @ self.Vk) @ self.Vk.T
            B = B - self.Uk @ (self.Uk.T @ B)
        return A, B

    def forward(self, x):
        A, B = self.effective()
        return self.base(x) + self.scale * ((self.drop(x) @ A.T) @ B.T)


def apply_lora(model, rank, alpha, dropout, protect_k):
    # replace each target linear and return the setup time for bases
    setup_seconds = 0.0
    idx = 0
    for layer in model.model.layers:
        for parent in [layer.self_attn, layer.mlp]:
            for name in LORA_TARGETS:
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


def lora_modules_of(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def export_adapters(model, path, meta):
    # raw and effective factors per module
    out = {"meta": meta, "modules": {}, "checks": {}}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            with torch.no_grad():
                A_eff, B_eff = m.effective()
                out["modules"][name] = {"A": m.A.detach().float().cpu(), "B": m.B.detach().float().cpu(),
                                        "A_eff": A_eff.detach().float().cpu(),
                                        "B_eff": B_eff.detach().float().cpu(), "scale": m.scale}
    # oplora leakage checks. both should be near zero.
    if meta.get("protect_k"):
        with torch.no_grad():
            leak_out = max(float((mod.Uk.T @ mod.effective()[1]).abs().max()) for mod in lora_modules_of(model))
            leak_in = max(float((mod.effective()[0] @ mod.Vk).abs().max()) for mod in lora_modules_of(model))
        out["checks"] = {"max_leak_out": leak_out, "max_leak_in": leak_in}
    torch.save(out, path)


# checkpoints

def trainable_state(named):
    return {n: p.detach().cpu() for n, p in named}


def rng_state():
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_recovery(path, step, named, opt, sched, scaler):
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "trainable": trainable_state(named), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng_state()}, tmp)
    os.replace(tmp, path)


def save_snapshot(path, method, model, tok, meta):
    path.mkdir(parents=True, exist_ok=True)
    if method in ("full", "l2sp"):
        sd = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
        if model.config.tie_word_embeddings:
            sd.pop("lm_head.weight", None)  # tied to the embedding, saved once
        model.save_pretrained(path, state_dict=sd)
        tok.save_pretrained(path)
        # fp32 down_proj copies for diagnostics
        fp32 = {L: model.model.layers[L].mlp.down_proj.weight.detach().float().cpu() for L in (0, 13, 27)}
        torch.save(fp32, path / "down_proj_fp32.pt")
    else:
        export_adapters(model, path / "adapters.pt", meta)


# main

def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if world > 1 else 0

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    else:
        device = torch.device("cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.set_device(device)
    amp = device.type in ("cuda", "mps")

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    lr = args.lr or (2e-5 if args.method in ("full", "l2sp") else 2e-4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    tok = AutoTokenizer.from_pretrained(args.model)
    examples = load_examples(tok, args.num_examples)
    total_updates = len(examples) // EFFECTIVE_BATCH
    snap_steps = args.snapshot_steps or ("625,1250" if args.method == "full" else "1250")
    ckpt_steps = {int(s) for s in args.checkpoint_steps.split(",")}
    snap_steps = {int(s) for s in snap_steps.split(",")}

    # model. full and l2sp keep fp32 master weights. lora and oplora freeze an fp16 backbone.
    dtype = torch.float32 if args.method in ("full", "l2sp") else getattr(torch, args.backbone_dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    meta = {"method": args.method, "rank": args.rank, "alpha": args.alpha, "protect_k": 0,
            "lr": lr, "targets": LORA_TARGETS}
    setup_seconds = 0.0
    if args.method in ("full", "l2sp"):
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        if args.method == "oplora":
            meta["protect_k"] = args.protect_k
        setup_seconds = apply_lora(model, args.rank, args.alpha, args.dropout,
                                   args.protect_k if args.method == "oplora" else 0)
        model.to(device)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    if world > 1:
        for p in params:
            dist.broadcast(p.data, src=0)
    n_trainable = sum(p.numel() for p in params)

    # anchor for full ft displacement and l2-sp penalty. kept on cpu to save gpu memory.
    anchor = None
    if args.method in ("full", "l2sp"):
        anchor = [p.detach().float().cpu().clone() for p in params]
    anchor_bytes = sum(a.numel() * a.element_size() for a in anchor) if anchor else 0

    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=use_cuda)
    sched = get_cosine_schedule_with_warmup(opt, WARMUP_UPDATES, total_updates)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type in ("cuda", "mps"), init_scale=2 ** 11)

    rec_dir = Path(args.recovery_dir) if args.recovery_dir else out
    rec_dir.mkdir(parents=True, exist_ok=True)
    recovery = rec_dir / "recovery.pt"
    start = 0
    if recovery.exists():
        ck = torch.load(recovery, map_location="cpu", weights_only=False)
        lookup = dict(named)
        with torch.no_grad():
            for n, v in ck["trainable"].items():
                lookup[n].copy_(v.to(device))
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        start = ck["step"]
        log(f"resumed from update {start}")
        del ck

    log_path = out / "train_log.jsonl"
    peak = 0
    snap_seconds = {}
    penalty_total = 0.0
    final_disp = None
    t_start = time.time()
    bar = tqdm(total=total_updates, initial=start, desc=args.method, disable=(rank != 0), dynamic_ncols=False, ncols=110)
    for step in range(start, total_updates):
        t0 = time.time()
        batch = examples[step * EFFECTIVE_BATCH: (step + 1) * EFFECTIVE_BATCH]
        total_tokens = sum(len(a) for _, a in batch)
        micro = [batch[i: i + MICRO_BATCH] for i in range(0, EFFECTIVE_BATCH, MICRO_BATCH)]

        sft_sum = torch.zeros((), device=device)
        for mb in micro[rank::world]:
            ids, att, lab = collate(mb, tok.pad_token_id, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
                mask = lab[:, 1:] != -100
                mb_loss = chunked_ce(model.lm_head, hidden[:, :-1][mask], lab[:, 1:][mask])
            scaler.scale(mb_loss / total_tokens).backward()
            sft_sum += mb_loss.detach()
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device))

        if world > 1:
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
            dist.all_reduce(sft_sum)

        scale_before = scaler.get_scale()
        scaler.unscale_(opt)

        # l2sp penalty added once per rank
        omega = 0.0
        penalty_seconds = 0.0
        if anchor is not None:
            t_pen = time.time()
            om = torch.zeros((), device=device)
            for p, a in zip(params, anchor):
                d = p.detach() - a.to(device, non_blocking=True)
                om += (d * d).sum()
                if args.method == "l2sp" and args.lam > 0:
                    p.grad.add_(d, alpha=args.lam)
            omega = 0.5 * om.item()
            penalty_seconds = time.time() - t_pen

        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        lr_used = opt.param_groups[0]["lr"]
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        opt.zero_grad(set_to_none=True)
        sched.step()

        sft = sft_sum.item() / total_tokens
        penalty_total += penalty_seconds
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = max(peak, torch.cuda.max_memory_allocated(device))
        elif device.type == "mps":
            peak = max(peak, torch.mps.current_allocated_memory())
        done = step + 1
        disp_after = None
        if anchor is not None and (done in snap_steps or done == total_updates):
            # displacement after the step
            with torch.no_grad():
                sq = sum(((p.detach() - a.to(device)) ** 2).sum().item() for p, a in zip(params, anchor))
            disp_after = math.sqrt(sq)
        rec = {"update": step + 1, "sft_loss": sft, "omega": omega,
               "lam_omega": args.lam * omega if args.method == "l2sp" else 0.0,
               "total_loss": sft + (args.lam * omega if args.method == "l2sp" else 0.0),
               "displacement": math.sqrt(2 * omega) if anchor is not None else None,
               "displacement_after": disp_after, "penalty_seconds": round(penalty_seconds, 3),
               "lr": lr_used, "grad_norm": float(grad_norm), "skipped": skipped,
               "seconds": round(time.time() - t0, 2), "peak_mem_GB": round(peak / 1e9, 2)}
        if disp_after is not None:
            final_disp = disp_after
        bar.update(1)
        bar.set_postfix(sft=f"{sft:.4f}", lr=f"{lr_used:.1e}", s_per_upd=rec["seconds"], mem_GB=rec["peak_mem_GB"])
        if rank == 0:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        stopping = args.stop_at is not None and done == args.stop_at
        if done in ckpt_steps or done == total_updates or stopping:
            if rank == 0:
                save_recovery(recovery, done, named, opt, sched, scaler)
                if done in snap_steps:
                    t_snap = time.time()
                    save_snapshot(out / f"snapshot_{done}", args.method, model, tok, meta)
                    snap_seconds[f"snapshot_{done}"] = round(time.time() - t_snap, 2)
            if world > 1:
                dist.barrier()
        if stopping:
            log(f"stopped at update {done}; rerun the same command to resume")
            break

    bar.close()
    # rank sync check over trainable parameters
    if world > 1:
        chk = torch.stack([p.detach().double().sum() for p in params]).sum()
        all_chk = [torch.zeros_like(chk) for _ in range(world)]
        dist.all_gather(all_chk, chk)
        log(f"ranks in sync: {all(torch.isclose(c, all_chk[0]) for c in all_chk)}")

    if rank == 0:
        def dir_bytes(d):
            return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        ckpt_bytes = {p.name: dir_bytes(p) for p in out.glob("snapshot_*")}
        summary = {"method": args.method, "trainable_params": n_trainable,
                   "train_seconds": round(time.time() - t_start, 1), "peak_mem_GB": round(peak / 1e9, 2),
                   "anchor_bytes": anchor_bytes, "setup_seconds": round(setup_seconds, 2),
                   "basis_bytes": (sum(m.Uk.numel() + m.Vk.numel() for m in lora_modules(model)) * 4
                                   if args.method == "oplora" else 0),
                   "recovery_bytes": recovery.stat().st_size if recovery.exists() else 0,
                   "snapshot_bytes": ckpt_bytes, "snapshot_seconds": snap_seconds,
                   "penalty_seconds_total": round(penalty_total, 2), "final_displacement": final_disp,
                   "world_size": world, "updates_run": total_updates - start}
        json.dump(summary, open(out / "summary.json", "w"), indent=2)
        log(json.dumps(summary))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
