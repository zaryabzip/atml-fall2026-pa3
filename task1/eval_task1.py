# lm eval runner for pa3 task 1 over every model variant and four benchmarks
# python eval_task1.py --variants variants.json --out eval --device cuda
# variants.json maps a name to a model path or hub id
# finished benchmark outputs are skipped on rerun so split sessions can resume

import argparse
import gc
import json
from pathlib import Path

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# benchmark settings from the spec

MAX_NEW = 1024
MPS_FRACTION = 0.75
BENCH = [
    {"name": "gsm8k", "tasks": ["gsm8k_cot"], "chat": True},
    {"name": "hellaswag", "tasks": ["hellaswag"], "chat": False},
    {"name": "boolq", "tasks": ["boolq"], "chat": False, "batch": 4},  # long passages, full vocab logits per position
    {"name": "arc_easy", "tasks": ["arc_easy"], "chat": False},
]


def load_model(path, args):
    # load on cpu in the target dtype, then move. casting while loading straight onto mps segfaulted
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, args.dtype))
    return model.to(args.device).eval()


def free(device):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def gsm8k_truncated(samples, tok):
    # capped responses re-encode near the cap
    # the margin of 16 tokens covers re-encoding differences
    flags = []
    for s in samples:
        n = len(tok(s["resps"][0][0], add_special_tokens=False).input_ids)
        flags.append(n >= MAX_NEW - 16)
    return flags


def done(bench, out_dir):
    return (out_dir / f"{bench['name']}_results.json").exists() and (out_dir / f"{bench['name']}_samples.jsonl").exists()


def run_one(name, model, tok, bench, args, out_dir):
    if done(bench, out_dir):
        print(f"skip {name} {bench['name']}", flush=True)
        return
    free(args.device)  # drop cached buffers from the previous benchmark
    gen = {"max_gen_toks": MAX_NEW, "do_sample": False} if bench["chat"] else None
    batch =int(args.gsm8k_batch if bench["chat"] else min(int(args.batch_size), bench.get("batch", 10**9)))
    while True:
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch,
                  enable_thinking=False if bench["chat"] else None)  # gsm8k uses the non thinking chat template
        try:
            res = lm_eval.simple_evaluate(
                model=lm,
                tasks=bench["tasks"],
                num_fewshot=0,
                apply_chat_template=bench["chat"],
                gen_kwargs=gen,
                log_samples=True,
                limit=args.limit,
            )
            break
        except RuntimeError as e:
            # out of memory on the device. halve the batch and retry
            if batch == 1 or not ("buffer size" in str(e) or "out of memory" in str(e)):
                raise
            del lm
            free(args.device)
            batch //= 2
            print(f"retry {name} {bench['name']} at batch {batch}: {e}", flush=True)
    task = bench["tasks"][0]
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = res.get("samples", {}).get(task, [])
    if args.limit is None and bench["chat"]:
        # one row per doc and filter
        per_filter = {}
        for s in samples:
            per_filter[s.get("filter", "none")] = per_filter.get(s.get("filter", "none"), 0) + 1
        assert all(v == 1319 for v in per_filter.values()), per_filter
    trunc = gsm8k_truncated(samples, tok) if bench["chat"] else None
    # per example file first, so a stop between the two writes is redone on rerun
    with open(out_dir / f"{bench['name']}_samples.jsonl", "w") as f:
        for i, s in enumerate(samples):
            row = {"doc_id": s.get("doc_id", i), "filter": s.get("filter"), "target": s.get("target"),
                   "prompt": s.get("arguments"),
                   "response": s["resps"][0][0] if s.get("resps") else None,
                   "filtered": s.get("filtered_resps"), "metrics": {k: v for k, v in s.items()
                                                                    if k in ("exact_match", "acc", "acc_norm")}}
            if trunc is not None:
                row["truncated"] = trunc[i]
            f.write(json.dumps(row, default=str) + "\n")
    (out_dir / f"{bench['name']}_results.json").write_text(json.dumps(
        {"variant": name, "bench": bench["name"], "metrics": res["results"][task],
         "limit": args.limit, "device": args.device, "dtype": args.dtype, "batch_size": lm.batch_size},
        indent=2, default=str))
    print(f"done {name} {bench['name']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch_size", default="8")
    ap.add_argument("--gsm8k_batch", default="32")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma separated variant names")
    args = ap.parse_args()
    if args.device == "mps":
        # the default mps limit is 1.7x the recommended working set, which spilled into swap on a 16 GB mac.
        # a hard cap raises out of memory instead and the batch is halved
        torch.mps.set_per_process_memory_fraction(MPS_FRACTION)
    variants = json.load(open(args.variants))
    keep = set(args.only.split(",")) if args.only else None
    out = Path(args.out)
    for name, path in variants.items():
        if keep and name not in keep:
            continue
        if all(done(b, out / name) for b in BENCH):
            print(f"skip {name}, all benchmarks done", flush=True)
            continue
        model = load_model(path, args)
        tok = AutoTokenizer.from_pretrained(path)
        failed = []
        for bench in BENCH:
            # one failed benchmark does not stop the rest
            try:
                run_one(name, model, tok, bench, args, out / name)
            except Exception as e:
                print(f"failed {name} {bench['name']}: {e!r}", flush=True)
                failed.append(bench["name"])
                free(args.device)
        del model
        free(args.device)
        if failed:
            raise SystemExit(f"failed benchmarks for {name}: {failed}")


if __name__ == "__main__":
    main()
