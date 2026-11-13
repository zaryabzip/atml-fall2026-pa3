# task 2 vlm training
# stage_scale is for smoke tests only

import argparse
import json
import os
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from vlm import ANSWER_CAP, CAPTION_CAP, VLM

# spec settings

SEED = 6304
ROOT = Path(__file__).resolve().parent.parent / "data" / "task2"
MAN = ROOT / "manifests"
FEATS = ROOT / "features"
MICRO, ACCUM = 8, 2
DEV_EVERY = 500
CKPT_EVERY = 250
# avoids early fp16 overflow skips
SCALER_INIT = 2 ** 11
# stage config
STAGES = [
    {"name": "caption", "updates": 2250, "conn_lr": 1e-3, "lora_lr": None,
     "mix": {"caption": 1.0}},
    {"name": "instruct", "updates": 1500, "conn_lr": 1e-4, "lora_lr": 2e-5,
     "mix": {"presence": 0.50, "counting": 0.25, "caption": 0.25}},
    {"name": "vqa", "updates": 750, "conn_lr": 5e-5, "lora_lr": 1e-5,
     "mix": {"vqa": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10}},
]


# data

def read_jsonl(name):
    return [json.loads(line) for line in open(MAN / name)]


class Features:
    # memmapped features
    def __init__(self):
        self.arr = np.load(FEATS / "features.npy", mmap_mode="r")
        self.index = json.load(open(FEATS / "index.json"))

    def get(self, image_ids):
        rows = [self.index[str(i)] for i in image_ids]
        return torch.from_numpy(np.stack([self.arr[r] for r in rows]))


def caption_example(rec, vlm):
    return {"image_id": rec["image_id"],
            "parts": vlm.build_example(None, "Describe the image.", rec["caption"], CAPTION_CAP)}


def qa_example(item, question, vlm):
    return {"image_id": item["image_id"],
            "parts": vlm.build_example(None, question, item["answer"], ANSWER_CAP)}


def pair_examples(pair, vlm):
    # pair shares question
    return [qa_example(it, pair["question"], vlm) for it in pair["items"]]


# schedules and rng

def lr_factor(local_step, total):
    # warmup then cosine decay
    warm = max(1, int(0.05 * total))
    if local_step < warm:
        return (local_step + 1) / warm
    progress = (local_step - warm) / max(1, total - warm)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def rng_state(device):
    st = {"torch": torch.get_rng_state()}
    if device.type == "mps":
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(device, st):
    # cpu tensors for rng setters
    torch.set_rng_state(st["torch"].cpu())
    if device.type == "mps" and "mps" in st:
        torch.mps.set_rng_state(st["mps"].cpu())


# sampling

class Sampler:
    # stage 1 walks captions
    def __init__(self, rng, data, vlm):
        self.rng = rng
        self.data = data
        self.vlm = vlm
        # order saved in checkpoint
        self.perm = []

    def caption_batch(self, n):
        caps = self.data["caption_train"]
        # new pass when order runs out
        if len(self.perm) < n:
            self.perm = [int(i) for i in self.rng.permutation(len(caps))]
        idx, self.perm = self.perm[:n], self.perm[n:]
        return [caption_example(caps[i], self.vlm) for i in idx], Counter({"caption": n})

    def draw_batch(self, mix, n_examples):
        keys = list(mix)
        probs = np.array([mix[k] for k in keys])
        out, counts = [], Counter()
        for _ in range(n_examples // 2):
            kind = keys[self.rng.choice(len(keys), p=probs)]
            counts[kind] += 1
            # caption replay draws two captions
            if kind == "caption":
                caps = self.data["caption_train"]
                for i in self.rng.choice(len(caps), size=2, replace=False):
                    out.append(caption_example(caps[i], self.vlm))
            else:
                pool = self.data[kind]
                pair = pool[self.rng.integers(len(pool))]
                out += pair_examples(pair, self.vlm)
        return out, counts


def make_batch(examples, feats, vlm):
    feat = feats.get([e["image_id"] for e in examples])
    return feat, [e["parts"] for e in examples]


# dev eval

def nll_rows(vlm, feats, image_ids, parts, with_image):
    feat = feats.get(image_ids) if with_image else None
    embeds, mask, labels = vlm.assemble(feat, parts, with_image=with_image)
    return vlm.example_nll(embeds, mask, labels).tolist()


@torch.no_grad()
def dev_eval(vlm, dev, feats):
    vlm.eval()
    out = {}
    # caption dev loss
    cap = [caption_example(r, vlm) for r in dev["caption_dev"]]
    total = 0.0
    for i in range(0, len(cap), MICRO):
        chunk = cap[i:i + MICRO]
        f, p = make_batch(chunk, feats, vlm)
        total += vlm.loss(f, p).item() * len(chunk)
    out["caption_dev_loss"] = total / len(cap)

    # answer nll on both dev suites
    # correct complementary no image
    # swap penalty positive favors correct
    for suite in ["grounding_dev", "vqa_dev"]:
        rows = []
        for pair in dev[suite]:
            a, b = pair["items"]
            for it, partner in ((a, b), (b, a)):
                prefix, suffix, a_ids = vlm.build_example(None, pair["question"], it["answer"], ANSWER_CAP)
                rows.append((it["image_id"], partner["image_id"], (prefix, suffix, a_ids[:-1])))
        sums = {"correct": 0.0, "complementary": 0.0, "no_image": 0.0}
        for i in range(0, len(rows), MICRO):
            chunk = rows[i:i + MICRO]
            parts = [r[2] for r in chunk]
            sums["correct"] += sum(nll_rows(vlm, feats, [r[0] for r in chunk], parts, True))
            sums["complementary"] += sum(nll_rows(vlm, feats, [r[1] for r in chunk], parts, True))
            sums["no_image"] += sum(nll_rows(vlm, feats, [r[0] for r in chunk], parts, False))
        n = len(rows)
        out[f"{suite}_nll_correct"] = sums["correct"] / n
        out[f"{suite}_nll_complementary"] = sums["complementary"] / n
        out[f"{suite}_nll_no_image"] = sums["no_image"] / n
        out[f"{suite}_swap_penalty"] = (sums["complementary"] - sums["correct"]) / n
    vlm.train()
    return out


def save_stage_file(out, idx, name, vlm):
    torch.save({"connector": vlm.connector.state_dict(), "lora": checkpoint_lora(vlm),
                "scale": vlm.scale, "scale_rule": vlm.scale_rule},
               out / f"stage{idx}_{name}.pt")


def checkpoint_lora(vlm):
    return {k: v for k, v in vlm.core.state_dict().items() if "lora_" in k}


# training loop

def main():
    # command line
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent.parent / "artifacts" / "task2" / "main"))
    ap.add_argument("--stage_scale", type=float, default=1.0)
    ap.add_argument("--ckpt_every", type=int, default=CKPT_EVERY)
    ap.add_argument("--stop_at", type=int, default=None,
                    help="save recovery state and exit after this many updates (split runs across sessions)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    stages = [dict(s, updates=max(1, int(s["updates"] * args.stage_scale))) for s in STAGES]
    total_main = sum(s["updates"] for s in stages)

    # load data and model
    feats = Features()
    vlm = VLM(device)
    data = {
        "caption_train": read_jsonl("captions_train.jsonl"),
        "presence": read_jsonl("presence_pool_train.jsonl"),
        "counting": read_jsonl("counting_pool_train.jsonl"),
        "vqa": read_jsonl("vqa_train.jsonl"),
    }
    dev = {
        "caption_dev": read_jsonl("captions_dev.jsonl"),
        "grounding_dev": read_jsonl("grounding_dev.jsonl"),
        "vqa_dev": read_jsonl("vqa_dev.jsonl"),
    }
    sampler = Sampler(rng, data, vlm)

    # frozen parameter check
    assert not [n for n, p in vlm.core.named_parameters() if p.requires_grad and "lora_" not in n], \
        "base LM parameters must stay frozen"

    # resume or calibrate
    start_step = 0
    resume_ck = None
    recovery = out / "recovery.pt"
    if recovery.exists():
        resume_ck = torch.load(recovery, map_location=device, weights_only=False)
        vlm.connector.load_state_dict(resume_ck["connector"])
        vlm.core.load_state_dict(resume_ck["lora"], strict=False)
        assert len(resume_ck["lora"]) == len(checkpoint_lora(vlm)), "LoRA keys do not match on resume"
        vlm.scale = resume_ck["scale"]
        vlm.scale_rule = resume_ck["scale_rule"]
        start_step = resume_ck["step"]
        rng.bit_generator.state = resume_ck["rng_state"]
        set_rng_state(device, resume_ck["torch_rng"])
        sampler.perm = resume_ck["perm"]
        print(f"resumed at update {start_step}", flush=True)
    else:
        # calibrate on 512 captions
        cal_sample = data["caption_train"][:512]
        cal_feats = feats.get([r["image_id"] for r in cal_sample])
        cal_text = torch.tensor([t for r in cal_sample for t in vlm.encode_text(r["caption"])[:CAPTION_CAP]])
        vlm.calibrate(cal_feats, cal_text)
        print(json.dumps({"calibration_scale": vlm.scale, "rule": vlm.scale_rule}), flush=True)

    # config once per fresh run
    if start_step == 0:
        config = {"seed": SEED, "args": vars(args), "stages": stages, "micro": MICRO, "accum": ACCUM,
                  "scaler_init_scale": SCALER_INIT, "device": str(device),
                  "scale": vlm.scale, "scale_rule": vlm.scale_rule}
        (out / "config.json").write_text(json.dumps(config, indent=2))

    # restore scaler on resume
    scaler = torch.amp.GradScaler(device.type, init_scale=SCALER_INIT)
    if resume_ck is not None:
        scaler.load_state_dict(resume_ck["scaler"])
    step = 0
    peak = 0
    t_run = time.time()
    dev_log = out / "dev_log.jsonl"
    train_log = out / "train_log.jsonl"
    # stage param counts
    stage_info = [{"stage": s["name"], "connector_params": sum(p.numel() for p in vlm.connector.parameters()),
                   "lora_params": 0 if i == 0 else sum(p.numel() for n, p in vlm.core.named_parameters()
                                                         if "lora_" in n)}
                  for i, s in enumerate(stages)]
    # connector grad check once
    checked_grads = False
    recovery_tmp = recovery.with_suffix(".tmp")

    # dev eval before training
    if start_step == 0:
        with torch.autocast(device.type, dtype=torch.float16):
            res = dev_eval(vlm, dev, feats)
        res["update"] = 0
        with open(dev_log, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)

    # stage loop
    for stage_idx, stage in enumerate(stages):
        stage_start = step
        stage_end = step + stage["updates"]
        # skip finished stages
        if stage_end <= start_step or step >= total_main:
            step = stage_end
            continue
        vlm.set_stage(stage_idx + 1)
        conn, lora = vlm.trainable(stage_idx + 1)
        if stage_idx == 0:
            assert not any(p.requires_grad for n, p in vlm.core.named_parameters() if "lora_" in n), \
                "LoRA must be off in stage 1"
        groups = [{"params": conn, "lr": stage["conn_lr"]}]
        if lora:
            groups.append({"params": lora, "lr": stage["lora_lr"]})
        # fresh adamw per stage
        opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.999))
        # restore opt only mid stage
        if resume_ck is not None and stage_start < start_step < stage_end:
            opt.load_state_dict(resume_ck["opt"])
            resume_ck = None
        base_lrs = [g["lr"] for g in opt.param_groups]
        n_conn = sum(p.numel() for p in conn)
        n_lora = sum(p.numel() for p in lora)
        print(f"stage {stage_idx + 1} ({stage['name']}): connector {n_conn} lora {n_lora}", flush=True)
        vlm.train()

        # update loop
        for u in range(stage["updates"]):
            main_step = stage_start + u
            if main_step >= total_main:
                break
            # skip updates before checkpoint
            if main_step < start_step:
                continue
            t0 = time.time()
            if stage["name"] == "caption":
                examples, counts = sampler.caption_batch(MICRO * ACCUM)
            else:
                examples, counts = sampler.draw_batch(stage["mix"], MICRO * ACCUM)
            f = lr_factor(u, stage["updates"])
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * f

            # accumulate micro batches
            total_loss = 0.0
            for m in range(ACCUM):
                chunk = examples[m * MICRO:(m + 1) * MICRO]
                fe, pa = make_batch(chunk, feats, vlm)
                with torch.autocast(device.type, dtype=torch.float16):
                    loss = vlm.loss(fe, pa) / ACCUM
                scaler.scale(loss).backward()
                if device.type == "mps":
                    peak = max(peak, torch.mps.current_allocated_memory())
                total_loss += loss.item()

            # connector grad check
            if not checked_grads:
                assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in conn), \
                    "connector receives no gradient"
                checked_grads = True

            # unscale before clip
            scale_before = scaler.get_scale()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(conn + lora, 1.0)
            scaler.step(opt)
            scaler.update()
            # inf step drops the scale
            skipped = scaler.get_scale() < scale_before
            opt.zero_grad(set_to_none=True)

            # memory sampled per update
            done = main_step + 1
            if device.type == "mps":
                peak = max(peak, torch.mps.current_allocated_memory())
            rec = {"update": done, "stage": stage["name"], "loss": total_loss, "mix_counts": dict(counts),
                   "lr_factor": f, "grad_norm": float(grad_norm), "skipped": skipped,
                   "seconds": round(time.time() - t0, 2), "mem_GB_sampled": round(peak / 1e9, 2)}
            with open(train_log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if done % 50 == 0:
                print(json.dumps(rec), flush=True)

            # dev eval
            if done % DEV_EVERY == 0 or done in (stage_end, total_main):
                with torch.autocast(device.type, dtype=torch.float16):
                    res = dev_eval(vlm, dev, feats)
                res["update"] = done
                with open(dev_log, "a") as fh:
                    fh.write(json.dumps(res) + "\n")
                print(json.dumps(res), flush=True)

            # recovery checkpoint
            stopping = args.stop_at is not None and done == args.stop_at
            if done % args.ckpt_every == 0 or done == total_main or stopping:
                torch.save({"step": done, "connector": vlm.connector.state_dict(),
                            "lora": checkpoint_lora(vlm), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "rng_state": rng.bit_generator.state,
                            "torch_rng": rng_state(device), "perm": sampler.perm,
                            "scale": vlm.scale, "scale_rule": vlm.scale_rule}, recovery_tmp)
                os.replace(recovery_tmp, recovery)
            if stopping:
                if done == stage_end and stage_end <= total_main:
                    save_stage_file(out, stage_idx + 1, stage["name"], vlm)
                print(f"stopped at update {done}; resume by rerunning the same command", flush=True)
                return

        # stage weights
        if stage_end <= total_main:
            save_stage_file(out, stage_idx + 1, stage["name"], vlm)
        step = stage_end
        if step >= total_main:
            break

    # summary
    summary = {"updates_run_total": total_main, "stages": stage_info,
               "wall_seconds": round(time.time() - t_run, 1),
               "peak_mem_GB_sampled": round(peak / 1e9, 2)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
