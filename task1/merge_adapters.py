# merges a lora or oplora adapter file into a fresh qwen3 copy and checks it against the unmerged model
# python merge_adapters.py --adapters artifacts/task1/oplora16/snapshot_1250/adapters.pt --out merged/oplora16

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model and check prompts

QWEN = "Qwen/Qwen3-0.6B"
CHECK_TEXTS = [
    "Q: Tom has 3 apples and buys 4 more. How many apples does he have now?\nA:",
    "The capital of France is",
    "def add(a, b):\n    return",
    "Once upon a time, a small fox",
]


def load_base(base, device, dtype):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    model.eval()
    return tok, model


def delta_for(mod, dev):
    # low rank update from the saved effective factors
    A = mod["A_eff"].to(dev, torch.float32)
    B = mod["B_eff"].to(dev, torch.float32)
    return mod["scale"] * (B @ A)


def unmerged_hooks(model, modules, dev):
    # adds the low rank path in forward so the base weights stay untouched
    handles = []
    for name, mod in modules.items():
        base = dict(model.named_modules())[name]
        A = mod["A_eff"].to(dev, torch.float32)
        B = mod["B_eff"].to(dev, torch.float32)
        scale = mod["scale"]

        def hook(m, inputs, output, A=A, B=B, scale=scale):
            x = inputs[0].float()
            return output + (scale * ((x @ A.T) @ B.T)).to(output.dtype)

        handles.append(base.register_forward_hook(hook))
    return handles


@torch.no_grad()
def logits_for(model, tok, dev):
    enc = tok(CHECK_TEXTS, return_tensors="pt", padding=True).to(dev)
    return model(**enc).logits.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--base", default=QWEN)
    args = ap.parse_args()
    dev = torch.device(args.device)
    data = torch.load(args.adapters, weights_only=False)
    modules = data["modules"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # unmerged reference in fp32
    tok, ref = load_base(args.base, dev, torch.float32)
    handles = unmerged_hooks(ref, modules, dev)
    ref_logits = logits_for(ref, tok, dev)
    for h in handles:
        h.remove()

    # merged copy in fp32 for the exact check
    _, merged = load_base(args.base, dev, torch.float32)
    mods = dict(merged.named_modules())
    with torch.no_grad():
        for name, mod in modules.items():
            mods[name].weight.add_(delta_for(mod, dev))
    merged_logits = logits_for(merged, tok, dev)
    max_abs = (merged_logits - ref_logits).abs().max().item()
    rel = max_abs / ref_logits.abs().max().item()

    # fp16 copy saved for eval
    merged.half()
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    report = {"max_abs_logit_diff_fp32": max_abs, "relative_diff_fp32": rel,
              "modules_merged": len(modules), "saved": str(out)}
    (out / "merge_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
