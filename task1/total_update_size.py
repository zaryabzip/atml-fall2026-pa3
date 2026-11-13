# total update size over every weight, per method: ||dW|| and ||dW||/||W0||, by module type
# python task1/total_update_size.py   writes results/task1/analysis/total_update_size.json
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

R = Path(__file__).resolve().parent.parent
V = json.loads((R / "artifacts/task1/eval_variants.json").read_text())


def load(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def kind(name):
    for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]:
        if k in name:
            return k
    return "norm_other"


base = load(V["baseline"])
out = {}
for v, path in V.items():
    if v == "baseline":
        continue
    sd = load(path)
    num, den = defaultdict(float), defaultdict(float)
    changed = total = 0
    for k, w0 in base.items():
        if k not in sd or k == "lm_head.weight":  # lm_head is tied to embed_tokens, count it once
            continue
        d = sd[k] - w0
        c = kind(k)
        num[c] += d.pow(2).sum().item()
        den[c] += w0.pow(2).sum().item()
        changed += (d != 0).sum().item()
        total += d.numel()
    tn, td = sum(num.values()), sum(den.values())
    out[v] = dict(total_norm=tn ** 0.5, total_rel=(tn / td) ** 0.5, frac_changed=changed / total,
                  by_kind={c: dict(norm=num[c] ** 0.5, rel=(num[c] / den[c]) ** 0.5) for c in num})
    print(f"{v:16s} ||dW|| {tn ** .5:7.3f}  rel {100 * (tn / td) ** .5:5.2f}%  changed {100 * changed / total:5.1f}%",
          {c: round(100 * (num[c] / den[c]) ** .5, 2) for c in ["q_proj", "o_proj", "gate_proj", "down_proj", "embed_tokens"] if c in num}, flush=True)
    del sd
(R / "results/task1/analysis/total_update_size.json").write_text(json.dumps(out, indent=1))
