# weight space and activation diagnostics for pa3 task 1
# layers 0 13 27 down_proj. rho_k and eta on the update. r(z) on a seeded probe set.
# python diagnostics_task1.py --full_mid snap625 --full_final snap1250 --l2sp_final ... --lora4 ... --lora16 ... --oplora16 ...

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# settings from the spec

SEED = 6304
LAYERS = [0, 13, 27]
N_PROBE = 32
PROTECT_K = 128
QWEN = "Qwen/Qwen3-0.6B"


def hs_preprocess(text):
    # same text cleanup as lm eval hellaswag utils
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def hs_query(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return hs_preprocess(doc["activity_label"] + ": " + ctx)


def build_probe(tok, rng):
    # 32 gsm8k and 32 hellaswag probes with seed 6304
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    gi = sorted(int(i) for i in rng.choice(len(gsm), N_PROBE, replace=False))
    hi = sorted(int(i) for i in rng.choice(len(hs), N_PROBE, replace=False))
    texts = {"math": [], "non_math": []}
    for i in gi:
        # gsm8k prompt in the non thinking chat template
        prompt = tok.apply_chat_template([{"role": "user", "content": f"Q: {gsm[i]['question']}\nA:"}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        texts["math"].append(prompt)
    for i in hi:
        # hellaswag uses plain completion text without a chat template
        texts["non_math"].append(hs_query(hs[i]))
    ids = {"math": gi, "non_math": hi}
    return texts, ids


@torch.no_grad()
def cache_inputs(model, tok, texts, device):
    # down_proj input at last prompt token
    cache = {}
    for dom, plist in texts.items():
        vals = {L: [] for L in LAYERS}
        for p in plist:
            enc = tok(p, return_tensors="pt", add_special_tokens=False).to(device)
            store = {}
            hooks = []
            for L in LAYERS:
                mod = model.model.layers[L].mlp.down_proj
                hooks.append(mod.register_forward_pre_hook(
                    lambda m, inp, L=L: store.__setitem__(L, inp[0].detach())))
            model(**enc)
            for h in hooks:
                h.remove()
            last = enc.input_ids.shape[1] - 1
            for L in LAYERS:
                vals[L].append(store[L][0, last].float().cpu())
        cache[dom] = {L: torch.stack(vals[L]) for L in LAYERS}
    return cache


def load_w0(base_model, L):
    return base_model.model.layers[L].mlp.down_proj.weight.detach().float().cpu()


def delta_from_snapshot(path, L, w0):
    # fp32 copies are saved with full ft snapshots to avoid fp16 rounding in the update
    fp32 = Path(path) / "down_proj_fp32.pt"
    if fp32.exists():
        w = torch.load(fp32, weights_only=False)[L].float()
    else:
        sd = load_file(str(Path(path) / "model.safetensors"))
        w = sd[f"model.layers.{L}.mlp.down_proj.weight"].float()
    return w - w0


def delta_from_adapters(data, L):
    m = data["modules"][f"model.layers.{L}.mlp.down_proj"]
    return m["scale"] * (m["B_eff"].float() @ m["A_eff"].float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=QWEN)
    ap.add_argument("--full_mid", required=True)
    ap.add_argument("--full_final", required=True)
    ap.add_argument("--l2sp_final", required=True)
    ap.add_argument("--lora4", required=True)
    ap.add_argument("--lora16", required=True)
    ap.add_argument("--oplora16", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    rng = np.random.default_rng(SEED)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(dev)
    base.eval()

    texts, ids = build_probe(tok, rng)
    (out / "probe_ids.json").write_text(json.dumps(ids))
    cache = cache_inputs(base, tok, texts, dev)
    w0 = {L: load_w0(base, L) for L in LAYERS}

    variants = {
        "full_mid": ("snapshot", args.full_mid),
        "full_final": ("snapshot", args.full_final),
        "l2sp_final": ("snapshot", args.l2sp_final),
        "lora4_final": ("adapters", args.lora4),
        "lora16_final": ("adapters", args.lora16),
        "oplora16_final": ("adapters", args.oplora16),
    }

    weight_rows, act_rows = [], []
    for name, (kind, path) in variants.items():
        data = torch.load(path, weights_only=False) if kind == "adapters" else None
        for L in LAYERS:
            if kind == "snapshot":
                dw = delta_from_snapshot(path, L, w0[L])
            else:
                dw = delta_from_adapters(data, L)
            W0 = w0[L]
            # protected output subspace from the starting weight
            # bases from fp16 values as in training
            U, _, _ = torch.linalg.svd(W0.half().float(), full_matrices=False)
            Uk = U[:, :PROTECT_K]
            rho = float((Uk.T @ dw).pow(2).sum() / dw.pow(2).sum().clamp(min=1e-30))
            eta = float(dw.norm() / W0.norm())
            weight_rows.append({"variant": name, "layer": L, "rho_k": rho, "eta": eta,
                                "delta_fro": float(dw.norm()), "w0_fro": float(W0.norm())})
            for dom in ["math", "non_math"]:
                z = cache[dom][L]
                num = (z @ dw.T).norm(dim=1)
                den = (z @ W0.T).norm(dim=1) + 1e-8
                for i, r in enumerate((num / den).tolist()):
                    act_rows.append({"variant": name, "layer": L, "domain": dom, "prompt": i, "R": r})
    wdf = pd.DataFrame(weight_rows)
    adf = pd.DataFrame(act_rows)
    wdf.to_csv(out / "weight_space.csv", index=False)
    adf.to_csv(out / "activation_per_prompt.csv", index=False)
    summary = adf.groupby(["variant", "layer", "domain"])["R"].agg(["mean", "median"]).reset_index()
    summary.to_csv(out / "activation_summary.csv", index=False)
    print(wdf.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
