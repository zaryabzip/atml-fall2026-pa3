# does math fine-tuning shift the yes/no prior? content-free BoolQ-format prompts, p(yes) vs p(no)
# python task1/boolq_prior_test.py   writes results/task1/analysis/boolq_prior_test2.json
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = Path(__file__).resolve().parent.parent
V = json.loads((R / "artifacts/task1/eval_variants.json").read_text())
# lm-eval boolq format: "{passage}\nQuestion: {question}?\nAnswer:" then " no" / " yes"
PROMPTS = {"N/A passage and question": "N/A\nQuestion: N/A?\nAnswer:",
           "empty passage, N/A question": "\nQuestion: N/A?\nAnswer:",
           "neutral question": "N/A\nQuestion: is it?\nAnswer:"}
dev = "mps" if torch.backends.mps.is_available() else "cpu"
out = {}
for name, path in V.items():
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(dev).eval()
    yes, no = tok(" yes", add_special_tokens=False).input_ids, tok(" no", add_special_tokens=False).input_ids
    assert len(yes) == 1 and len(no) == 1
    res = {}
    for k, p in PROMPTS.items():
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            lp = torch.log_softmax(m(ids).logits[0, -1].float(), -1)
        d = (lp[yes[0]] - lp[no[0]]).item()
        res[k] = dict(logit_yes_minus_no=d, p_yes_of_two=torch.sigmoid(torch.tensor(d)).item(),
                      mass_yes_no=(lp[yes[0]].exp() + lp[no[0]].exp()).item())
    out[name] = res
    print(name, {k: round(v["logit_yes_minus_no"], 2) for k, v in res.items()}, flush=True)
    del m
    if dev == "mps":
        torch.mps.empty_cache()
(R / "results/task1/analysis/boolq_prior_test2.json").write_text(json.dumps(out, indent=1))
