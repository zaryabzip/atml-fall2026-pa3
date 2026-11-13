# held-out metamathqa evaluation for task 1, an extra experiment outside the spec
# compares each model's gain on unseen metamathqa questions with its gain on gsm8k, to see whether the gain follows
# the training distribution and format or carries over to other math questions
# the sample matches the training type mix, shares no original question with training, and is fixed before generation
# python task1/heldout_task1.py select
# python task1/heldout_task1.py run --only baseline
# python task1/heldout_task1.py report

import argparse
import gc
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "task3"))
import answers  # noqa: E402
from tabkit import TODO_CAPTION, num, table  # noqa: E402

RES = R / "results" / "task1" / "heldout"
ART = R / "artifacts" / "task1" / "heldout"
EVAL = R / "artifacts" / "task1" / "eval"
TABLES = R / "report" / "tables"
VARIANTS = json.load(open(R / "artifacts" / "task1" / "eval_variants.json"))
NAMES = {"baseline": "Baseline", "full_mid": "Full FT, mid", "full_final": "Full FT, final", "l2sp_final": "L2-SP, final",
         "lora4_final": "LoRA-4, final", "lora16_final": "LoRA-16, final", "oplora16_final": "OPLoRA-16, final"}

SEED = 6304
SAMPLE = 1000
MAX_LEN = 2048  # same limit as training
MAX_NEW = 1024  # same generation settings as gsm8k
BATCH = 16
MPS_FRACTION = 0.75
BOOT = 1000
STOPS = ["Q:", "</s>", "<|im_end|>"]
ANSWER = "The answer is:"


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


# sample, fixed before any generation

def type_sizes(types):
    # largest remainder, so the sample adds up exactly and follows the training mix
    c = Counter(types)
    exact = {t: SAMPLE * n / len(types) for t, n in c.items()}
    sizes = {t: int(x) for t, x in exact.items()}
    for t in sorted(exact, key=lambda t: sizes[t] - exact[t])[: SAMPLE - sum(sizes.values())]:
        sizes[t] += 1
    return sizes


def select():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    gsm8k = {normalize(q) for q in load_dataset("openai/gsm8k", "main", split="test")["question"]}
    rows = json.load(open(hf_hub_download("meta-math/MetaMathQA", "MetaMathQA-395K.json", repo_type="dataset")))
    ids = json.load(open(R / "results" / "task1" / "lora_kaggle" / "train_examples.json"))["row_ids"]
    train_q = {normalize(rows[i]["query"]) for i in ids}
    train_o = {normalize(rows[i]["original_question"]) for i in ids}
    sizes = type_sizes([rows[i]["type"] for i in ids])
    by_type, counts = {}, Counter()
    for i in range(max(ids) + 1, len(rows)):
        r = rows[i]
        counts["after_training"] += 1
        q, o = normalize(r["query"]), normalize(r["original_question"])
        if not r["query"].strip() or not r["response"].strip() or ANSWER not in r["response"]:
            counts["no_answer_line"] += 1
        elif q in gsm8k or o in gsm8k:
            counts["gsm8k_test_overlap"] += 1
        elif o in train_o or q in train_q:
            counts["shares_question_with_training"] += 1
        else:
            by_type.setdefault(r["type"], {}).setdefault(q, []).append(i)
    rng = np.random.default_rng(SEED)
    picked = []
    for t in sorted(sizes):
        n = 0
        for q in rng.permutation(sorted(by_type[t])):
            cands = by_type[t][str(q)]
            i = int(cands[rng.integers(len(cands))])
            r = rows[i]
            prompt = tok.apply_chat_template([{"role": "user", "content": r["query"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            length = len(tok(prompt, add_special_tokens=False).input_ids) + len(tok(r["response"] + "<|im_end|>", add_special_tokens=False).input_ids)
            if length > MAX_LEN:
                counts["too_long"] += 1
                continue
            picked.append({"row": i, "query": r["query"], "response": r["response"], "type": t,
                           "original": normalize(r["original_question"]), "gold": r["response"].rsplit(ANSWER, 1)[1].strip()})
            n += 1
            if n == sizes[t]:
                break
        assert n == sizes[t], (t, n, sizes[t])
    RES.mkdir(parents=True, exist_ok=True)
    info = {"seed": SEED, "training_rows": len(ids), "last_training_row": max(ids), "filter_counts": dict(counts),
            "distinct_clean_queries_by_type": {t: len(v) for t, v in sorted(by_type.items())},
            "sample_by_type": sizes, "sampled": len(picked),
            "rule": "rows after training, no shared original question or query with training, no gsm8k test overlap, "
                    "training type mix, one row per distinct query, seeded permutation of sorted queries"}
    (RES / "selection.json").write_text(json.dumps({"info": info, "rows": picked}, indent=1))
    print(json.dumps(info, indent=1))


def load_selection():
    return json.load(open(RES / "selection.json"))["rows"]


# scoring, the same for every model and for both benchmarks. it does not depend on the answer format

GSM_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def last_number(text):
    # the stock gsm8k_cot flexible-extract pattern
    m = GSM_PATTERN.findall(text)
    g = [x for x in m[-1] if x] if m else []
    return g[0].strip().strip(",.$") if g else None


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = k = i + len("\\boxed{")
    depth = 1
    while k < len(text):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k].strip() or None
        k += 1
    return None


def extract(text):
    # last box, then the text after the last answer line, then the last number. returns the answer and the path used
    b = last_boxed(text)
    if b:
        return b, "boxed"
    if ANSWER in text:
        a = text.rsplit(ANSWER, 1)[1].strip().rstrip(".")
        if a:
            return a, "answer_line"
    n = last_number(text)
    return (n, "last_number") if n else (None, "none")


def score(text, gold):
    a, path = extract(text)
    return (float(answers.math_equiv(gold, a)) if a else 0.0), path


def has_format(text):
    return ANSWER in text.rstrip().rsplit("\n", 1)[-1]


# generation and reference likelihood

def prompt_of(tok, query):
    return tok.apply_chat_template([{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def free(device):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def generate_all(model, tok, rows, path, device, max_new):
    done = {json.loads(l)["row"] for l in open(path)} if path.exists() else set()
    todo = sorted([r for r in rows if r["row"] not in done], key=lambda r: -len(r["query"]))
    with open(path, "a") as f:
        for b in range(0, len(todo), BATCH):
            chunk = todo[b:b + BATCH]
            enc = tok([prompt_of(tok, r["query"]) for r in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, use_cache=True, stop_strings=STOPS,
                                 tokenizer=tok, pad_token_id=tok.pad_token_id)
            new = out[:, enc.input_ids.shape[1]:]
            for r, ids in zip(chunk, new):
                n = int((ids != tok.pad_token_id).sum())
                text = tok.decode(ids, skip_special_tokens=True)
                for s in STOPS:
                    text = text.split(s)[0]
                f.write(json.dumps({"row": r["row"], "response": text, "new_tokens": n, "truncated": n >= max_new}) + "\n")
            f.flush()
            print(f"{min(b + BATCH, len(todo))}/{len(todo)}", flush=True)


@torch.no_grad()
def reference_nll(model, tok, rows, path, device):
    # per-token nll of the reference solution: the solution body, the literal answer phrase and the answer value
    done = {json.loads(l)["row"] for l in open(path)} if path.exists() else set()
    with open(path, "a") as f:
        for r in rows:
            if r["row"] in done:
                continue
            p = tok(prompt_of(tok, r["query"]), add_special_tokens=False).input_ids
            enc = tok(r["response"] + "<|im_end|>", add_special_tokens=False, return_offsets_mapping=True)
            ids = torch.tensor([p + enc.input_ids], device=device)
            hidden = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:-1]
            nll = torch.nn.functional.cross_entropy(model.lm_head(hidden).float(), ids[0, len(p):], reduction="none").cpu().numpy()
            start = r["response"].rfind(ANSWER)
            seg = {"body": [], "phrase": [], "value": []}
            for t, (a, _) in enumerate(enc.offset_mapping[:-1]):
                seg["body" if a < start else "phrase" if a < start + len(ANSWER) else "value"].append(float(nll[t]))
            f.write(json.dumps({"row": r["row"], "tokens": len(nll), "sum_all": float(nll.sum()),
                                **{k: float(np.mean(v)) if v else None for k, v in seg.items()}}) + "\n")


def run(args):
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(MPS_FRACTION)
    rows = load_selection()[: args.limit] if args.limit else load_selection()
    base = Path(args.out)
    for v in [args.only] if args.only else VARIANTS:
        out = base / v
        out.mkdir(parents=True, exist_ok=True)
        path = VARIANTS[v]
        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, args.dtype)).to(args.device).eval()
        generate_all(model, tok, rows, out / "answers.jsonl", args.device, args.max_new)
        free(args.device)
        reference_nll(model, tok, rows, out / "nll.jsonl", args.device)
        del model
        free(args.device)
        print(f"done {v}", flush=True)


# report

def paired_boot(a, b, rng, groups=None):
    # mean paired difference in points with a 95% interval. rows from the same original question are resampled together
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if groups is None:
        groups = list(range(len(d)))
    keys = sorted(set(groups))
    pos = {k: j for j, k in enumerate(keys)}
    g = np.array([pos[k] for k in groups])
    total, size = np.bincount(g, weights=d, minlength=len(keys)), np.bincount(g, minlength=len(keys))
    idx = rng.integers(0, len(keys), (BOOT, len(keys)))
    lo, hi = np.percentile(total[idx].sum(axis=1) / size[idx].sum(axis=1), [2.5, 97.5])
    return [float(100 * d.mean()), float(100 * lo), float(100 * hi)]


def report():
    sel = load_selection()
    gold = {r["row"]: r["gold"] for r in sel}
    original = [r["original"] for r in sorted(sel, key=lambda r: r["row"])]
    variants = [v for v in VARIANTS if (ART / v / "nll.jsonl").exists()]
    rng = np.random.default_rng(SEED)
    held, gsm_official, gsm_same, summary = {}, {}, {}, {}
    for v in variants:
        rows = {json.loads(l)["row"]: json.loads(l) for l in open(ART / v / "answers.jsonl")}
        assert set(rows) == set(gold), v
        ids = sorted(rows)
        got = [score(rows[i]["response"], gold[i]) for i in ids]
        held[v] = [g[0] for g in got]
        nll = [json.loads(l) for l in open(ART / v / "nll.jsonl")]
        gs = [json.loads(l) for l in open(EVAL / v / "gsm8k_samples.jsonl")]
        gs = sorted([r for r in gs if r["filter"] == "flexible-extract"], key=lambda r: r["doc_id"])
        gsm_official[v] = [float(r["metrics"]["exact_match"]) for r in gs]
        gsm_same[v] = [score(r["response"], str(r["target"]))[0] for r in gs]
        summary[v] = {
            "heldout_accuracy": 100 * float(np.mean(held[v])), "gsm8k_official": 100 * float(np.mean(gsm_official[v])),
            "gsm8k_same_scorer": 100 * float(np.mean(gsm_same[v])), "extraction": dict(Counter(g[1] for g in got)),
            "answer_line": 100 * float(np.mean([has_format(rows[i]["response"]) for i in ids])),
            "gsm8k_answer_line": 100 * float(np.mean([has_format(r["response"]) for r in gs])),
            "truncated": 100 * float(np.mean([rows[i]["truncated"] for i in ids])),
            "mean_new_tokens": float(np.mean([rows[i]["new_tokens"] for i in ids])),
            "nll_pooled": float(sum(x["sum_all"] for x in nll) / sum(x["tokens"] for x in nll)),
            **{f"nll_{k}": float(np.mean([x[k] for x in nll if x[k] is not None])) for k in ["body", "phrase", "value"]}}
    for v in variants:
        if v != "baseline":
            summary[v]["gain_heldout"] = paired_boot(held["baseline"], held[v], rng, original)
            summary[v]["gain_gsm8k_official"] = paired_boot(gsm_official["baseline"], gsm_official[v], rng)
            summary[v]["gain_gsm8k_same_scorer"] = paired_boot(gsm_same["baseline"], gsm_same[v], rng)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    gain = lambda v, k: "--" if v == "baseline" else f"{num(summary[v][k][0], sign=True)} [{num(summary[v][k][1], sign=True)}, {num(summary[v][k][2], sign=True)}]"
    rows = [[NAMES[v], num(summary[v]["gsm8k_official"]), gain(v, "gain_gsm8k_official"), num(summary[v]["gsm8k_same_scorer"]),
             gain(v, "gain_gsm8k_same_scorer"), num(summary[v]["heldout_accuracy"]), gain(v, "gain_heldout")] for v in variants]
    TABLES.joinpath("t1_heldout.tex").write_text(table(
        TODO_CAPTION, "tab:t1_heldout", ["Condition"] + ["Score", "Gain (pp)"] * 3, rows, "l" + "rr" * 3,
        top=[("", 1), ("GSM8K, stock scorer", 2), ("GSM8K, same scorer", 2), ("Held-out MetaMathQA", 2)],
        size=r"\setlength{\tabcolsep}{4pt}"))
    rows = [[NAMES[v], num(summary[v]["gsm8k_answer_line"]), num(summary[v]["answer_line"]), num(summary[v]["truncated"]),
             num(summary[v]["mean_new_tokens"], ".0f"), num(summary[v]["nll_pooled"], ".3f"), num(summary[v]["nll_body"], ".3f"),
             num(summary[v]["nll_phrase"], ".3f"), num(summary[v]["nll_value"], ".3f")] for v in variants]
    TABLES.joinpath("t1_heldout_format.tex").write_text(table(
        TODO_CAPTION, "tab:t1_heldout_format",
        ["Condition", "GSM8K", "Held-out", r"At cap\\(\%)", r"Mean\\tokens", "All", "Body", "Phrase", "Value"], rows, "l" + "r" * 8,
        top=[("", 1), (r"Answer line (\%)", 2), ("Held-out answers", 2), ("Reference NLL per token", 4)]))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "run", "report"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--limit", type=int, default=None, help="questions, for tests only")
    ap.add_argument("--max_new", type=int, default=MAX_NEW, help="token cap, lower for tests only")
    args = ap.parse_args()
    select() if args.mode == "select" else run(args) if args.mode == "run" else report()


if __name__ == "__main__":
    main()
