# task 3 step 4: live comparison of one long response, three parallel responses and sequential refinement
# on the 50 fixed math questions. fp16, batch size 1, every call independently seeded and saved as it finishes.
# python task3/live.py run --out artifacts/task3/live --device mps
# python task3/live.py summarize --out artifacts/task3/live --results results/task3/live

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import answers
import common

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"  # pinned in the supplied README.MD
SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}
CAPS = {"parallel": 1024, "single": 3072, "revision": 2048}


def last_boxed(text):
    # readme rule: last literal \boxed{ with balanced braces. absent, empty or unclosed gives none
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = k = i + len("\\boxed{")
    depth = 1
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j:k].strip() or None
        k += 1
    return None


def questions():
    gen = pq.read_table(common.DATA / "qwen25-math" / "generations.parquet", columns=["sample_id", "question_text"])
    text = dict(zip(gen.column("sample_id").to_pylist(), gen.column("question_text").to_pylist()))
    live = common.splits(set(text))["live_math"]
    return [(q, text[q]) for q in live]


def calls_for(qi):
    # five calls per question, each with its own seed. the first parallel response is the initial response
    base = common.SEED * 10000 + qi * 10
    return [("parallel_0", base + 0), ("parallel_1", base + 1), ("parallel_2", base + 2),
            ("single", base + 3), ("revision", base + 4)]


def generate(model, tok, messages, cap, seed, device):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cap, pad_token_id=tok.eos_token_id, **SAMPLING)
    if device.type == "mps":
        torch.mps.synchronize()
    secs = time.time() - t0
    new = out[0, enc.input_ids.shape[1]:]
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    ended = len(new) > 0 and int(new[-1]) in eos
    text = tok.decode(new, skip_special_tokens=True)
    peak = torch.mps.driver_allocated_memory() if device.type == "mps" else None
    return {"prompt": prompt, "response": text, "input_tokens": int(enc.input_ids.shape[1]),
            "output_tokens": int(len(new)), "truncated": bool(len(new) >= cap and not ended),
            "seconds": round(secs, 3), "driver_memory_bytes": peak, "extracted": last_boxed(text)}


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "calls.jsonl"
    done = {}
    if log.exists():
        for l in open(log):
            r = json.loads(l)
            done[(r["question"], r["call"])] = r
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    t_load = time.time()
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype)  # load on cpu first
    model = model.to(device).eval()
    load_secs = time.time() - t_load
    qs = questions()[: args.limit] if args.limit else questions()
    scale = args.cap_scale
    with open(log, "a") as f:
        for qi, (q, problem) in enumerate(qs):
            user = problem + "\n\n" + INSTRUCTION
            base_msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            for call, seed in calls_for(qi):
                if (q, call) in done:
                    continue
                if call == "revision":
                    initial = done[(q, "parallel_0")]["response"]
                    msgs = base_msgs + [{"role": "assistant", "content": initial}, {"role": "user", "content": REVIEW}]
                    cap = CAPS["revision"] // scale
                else:
                    msgs = base_msgs
                    cap = (CAPS["single"] if call == "single" else CAPS["parallel"]) // scale
                r = generate(model, tok, msgs, cap, seed, device)
                r.update({"question": q, "call": call, "seed": seed, "cap": cap})
                f.write(json.dumps(r) + "\n")
                f.flush()
                done[(q, call)] = r
                print(f"{qi + 1}/{len(qs)} {q} {call}: {r['output_tokens']} tokens in {r['seconds']:.1f} s"
                      f" boxed={r['extracted'] is not None} truncated={r['truncated']}", flush=True)
    (out / "run_config.json").write_text(json.dumps({"model": args.model, "revision": args.revision,
                                                     "dtype": args.dtype, "device": args.device, "caps": CAPS,
                                                     "cap_scale": scale, "sampling": SAMPLING, "system": SYSTEM,
                                                     "instruction": INSTRUCTION, "review": REVIEW,
                                                     "model_load_seconds": round(load_secs, 1),
                                                     "torch": torch.__version__}, indent=2))


def summarize(args):
    out, res = Path(args.out), Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(out / "calls.jsonl")]
    by = {}
    for r in rows:
        by.setdefault(r["question"], {})[r["call"]] = r
    gold = common.load_gold("math")
    per_q, stats = [], {s: {"correct": 0, "in": 0, "out": 0, "secs": 0.0, "truncated": 0, "no_box": 0, "n": 0}
                        for s in ["single", "parallel", "refinement"]}
    trans = {"wrong_to_correct": 0, "correct_to_wrong": 0, "same_correct": 0, "same_wrong": 0}
    for q, c in by.items():
        if not all(k in c for k in ["parallel_0", "parallel_1", "parallel_2", "single", "revision"]):
            continue
        par = [c[f"parallel_{i}"] for i in range(3)]
        labels = answers.group_math([p["extracted"] for p in par])
        counts, first = {}, {}
        for pos, l in enumerate(labels):
            counts[l] = counts.get(l, 0) + 1
            first.setdefault(l, pos)
        sc = max(counts, key=lambda l: (counts[l], -first[l]))
        ok = {"single": answers.grade_math(c["single"]["extracted"] or answers.NO_ANSWER, gold[q]),
              "parallel": sc != answers.NO_ANSWER and answers.grade_math(sc, gold[q]),
              "refinement": answers.grade_math(c["revision"]["extracted"] or answers.NO_ANSWER, gold[q])}
        init_ok = answers.grade_math(par[0]["extracted"] or answers.NO_ANSWER, gold[q])
        if init_ok == ok["refinement"]:
            trans["same_correct" if init_ok else "same_wrong"] += 1
        else:
            trans["wrong_to_correct" if ok["refinement"] else "correct_to_wrong"] += 1
        cost = {"single": [c["single"]], "parallel": par, "refinement": [par[0], c["revision"]]}
        for s, calls in cost.items():
            st = stats[s]
            st["n"] += 1
            st["correct"] += ok[s]
            st["in"] += sum(x["input_tokens"] for x in calls)
            st["out"] += sum(x["output_tokens"] for x in calls)
            st["secs"] += sum(x["seconds"] for x in calls)
            st["truncated"] += sum(x["truncated"] for x in calls)
            st["no_box"] += sum(x["extracted"] is None for x in calls)
        per_q.append({"question": q, **{f"{s}_correct": ok[s] for s in ok}, "initial_correct": init_ok,
                      "parallel_answers": [p["extracted"] for p in par], "parallel_selected": sc,
                      "single_answer": c["single"]["extracted"], "revised_answer": c["revision"]["extracted"]})
    calls_per = {"single": 1, "parallel": 3, "refinement": 2}
    summary = {}
    for s, st in stats.items():
        n = max(st["n"], 1)
        summary[s] = {"questions": st["n"], "accuracy": st["correct"] / n, "mean_input_tokens": st["in"] / n,
                      "mean_output_tokens": st["out"] / n, "mean_seconds": st["secs"] / n,
                      "truncation_rate_per_call": st["truncated"] / (n * calls_per[s]),
                      "no_boxed_rate_per_call": st["no_box"] / (n * calls_per[s])}
    peak = max((r["driver_memory_bytes"] or 0) for r in rows)
    (res / "live_summary.json").write_text(json.dumps({"strategies": summary, "revision_transitions": trans,
                                                      "peak_driver_memory_gb": round(peak / 1e9, 2),
                                                      "calls": len(rows)}, indent=2))
    (res / "live_per_question.json").write_text(json.dumps(per_q, indent=1))
    print(json.dumps(summary, indent=2), json.dumps(trans))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summarize"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cap_scale", type=int, default=1, help="divide every cap by this, for tests only")
    args = ap.parse_args()
    run(args) if args.mode == "run" else summarize(args)


if __name__ == "__main__":
    main()
