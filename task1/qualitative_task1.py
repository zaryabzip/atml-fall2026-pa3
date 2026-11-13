# qualitative generations for pa3 task 1
# 24 prompts times 7 variants greedy with thinking disabled and a 512 token cap
# also writes the 42 response annotation subset for the first and last underlying questions
# python qualitative_task1.py --prompts first_plot_questions.yaml --variants variants.json --out qualitative

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# settings from the spec

MAX_NEW = 512
BATCH = 8
FORMATS = ["plain", "json", "template"]


def load_prompts(path):
    # first_plot_questions.yaml: one paraphrase per entry. the format comes from the id suffix
    # (_json or _template, otherwise plain) and json entries carry their own system message
    items = yaml.safe_load(open(path))
    rows = []
    for it in items:
        qid, fmt = it["id"], "plain"
        for f in ["json", "template"]:
            if qid.endswith("_" + f):
                qid, fmt = qid[: -len(f) - 1], f
        assert len(it["paraphrases"]) == 1, it["id"]
        rows.append({"question_id": qid, "format": fmt, "prompt": it["paraphrases"][0],
                     "system": it.get("system"), "yaml_id": it["id"]})
    # checks the layout assumed above
    assert len(rows) == 24, f"expected 24 prompts, got {len(rows)}"
    for f in FORMATS:
        ids = [r["question_id"] for r in rows if r["format"] == f]
        assert len(ids) == 8 and len(set(ids)) == 8, f"format {f} needs 8 distinct ids"
    return rows


def annotation_ids(rows):
    # first and last plain question ids
    plain = [r["question_id"] for r in rows if r["format"] == "plain"]
    return [plain[0], plain[-1]]


@torch.no_grad()
def generate(path, rows, device, dtype):
    tok = AutoTokenizer.from_pretrained(path if Path(path).exists() else "Qwen/Qwen3-0.6B")
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    model.eval()
    outs = []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        texts = [tok.apply_chat_template(([{"role": "system", "content": r["system"]}] if r["system"] else [])
                                         + [{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        for r, ids in zip(chunk, new):
            n = int((ids != tok.pad_token_id).sum())
            outs.append({"response": tok.decode(ids, skip_special_tokens=True).strip(),
                         "truncated": n >= MAX_NEW})
    del model
    gc.collect()
    if str(device) == "mps":
        torch.mps.empty_cache()
    elif str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    rows = load_prompts(args.prompts)
    variants = json.load(open(args.variants))
    assert len(rows) == 24, f"expected 24 prompts, got {len(rows)}"

    records = []
    for name, path in variants.items():
        gens = generate(path, rows, args.device, dtype)
        for r, g in zip(rows, gens):
            records.append({"variant": name, **r, **g})
        print(f"done {name}", flush=True)
    # sort by question format variant
    qorder = list(dict.fromkeys(r["question_id"] for r in rows))
    vorder = list(variants)
    records.sort(key=lambda r: (qorder.index(r["question_id"]), FORMATS.index(r["format"]), vorder.index(r["variant"])))
    (out / "responses.json").write_text(json.dumps(records, indent=2))

    keep = set(annotation_ids(rows))
    fields = ["variant", "question_id", "format", "prompt", "response", "truncated",
              "format_compliance", "coherence", "repetition", "unnecessary_math",
              "inappropriate", "excerpt"]
    with open(out / "annotation_subset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            if rec["question_id"] in keep:
                w.writerow({k: rec.get(k, "") for k in fields})
    print(f"annotation subset rows: {sum(1 for r in records if r['question_id'] in keep)}", flush=True)


if __name__ == "__main__":
    main()
