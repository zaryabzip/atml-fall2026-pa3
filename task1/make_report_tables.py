# task 1 latex tables from the evaluation, diagnostics and analysis outputs
# run task1/analysis_task1.py first, it writes the csv files read here
# python task1/make_report_tables.py --eval artifacts/task1/eval

import argparse
import csv
import json
import math
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from tabkit import TODO_CAPTION, ci, num, table  # noqa: E402

OUT = R / "report" / "tables"
DIAG = R / "results" / "task1" / "diagnostics"

VARIANTS = [("baseline", "Baseline"), ("full_mid", "Full FT, mid"), ("full_final", "Full FT, final"),
            ("l2sp_final", "L2-SP, final"), ("lora4_final", "LoRA-4, final"),
            ("lora16_final", "LoRA-16, final"), ("oplora16_final", "OPLoRA-16, final")]
NAME = dict(VARIANTS)
PRIMARY = [("gsm8k", "exact_match", "flexible-extract", "GSM8K"), ("hellaswag", "acc_norm", "none", "HellaSwag"),
           ("boolq", "acc", "none", "BoolQ"), ("arc_easy", "acc_norm", "none", "ARC-Easy")]
COMPARE = [("Rank", "lora4_final", "lora16_final", r"LoRA-4 $\to$ LoRA-16"),
           ("Regularization", "full_final", "l2sp_final", r"Full FT $\to$ L2-SP"),
           ("Protection", "lora16_final", "oplora16_final", r"LoRA-16 $\to$ OPLoRA-16"),
           ("Method", "full_final", "lora16_final", r"Full FT $\to$ LoRA-16")]


# io

def result(ev, v, bench):
    f = ev / v / f"{bench}_results.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return None if d.get("limit") else d["metrics"]


def metric(ev, v, bench, key, filt):
    m = result(ev, v, bench)
    return None if m is None else m.get(f"{key},{filt}")


def per_example(ev, v, bench, key, filt):
    # doc id to score from the samples file
    f = ev / v / f"{bench}_samples.jsonl"
    if result(ev, v, bench) is None or not f.exists():
        return None
    out = {}
    for line in open(f):
        r = json.loads(line)
        if r["filter"] == filt:
            out[r["doc_id"]] = float(r["metrics"][key])
    return out


def pct(x):
    return "TODO" if x is None else num(100 * x)


# tables

def benchmark(ev):
    rows = []
    base = {b: metric(ev, "baseline", b, k, f) for b, k, f, _ in PRIMARY}
    for v, name in VARIANTS:
        row = [name]
        for b, k, f, _ in PRIMARY:
            x = metric(ev, v, b, k, f)
            se = metric(ev, v, b, k + "_stderr", f)
            row.append("TODO" if x is None else f"{num(100 * x)} $\\pm$ {num(100 * se)}")
            row.append("--" if v == "baseline" else
                       ("TODO" if x is None or base[b] is None else num(100 * (x - base[b]), sign=True)))
        rows.append(row)
    OUT.joinpath("t1_benchmark.tex").write_text(table(
        "Benchmark scores (\\%, with standard error) for the seven evaluated models. For GSM8K we used flexible-extraction exact match, for HellaSwag and ARC-Easy the normalized accuracy of the likelihoods, and for BoolQ the accuracy of the likelihoods. Delta columns report the change in percentage points from the baseline.",
        "tab:t1_benchmark", ["Condition"] + ["Score", r"$\Delta$"] * 4, rows, "l" + "rr" * 4,
        top=[("", 1)] + [(t, 2) for *_, t in PRIMARY]))


def comparisons(ev):
    rows = []
    for q, a, b, name in COMPARE:
        row = [q, name]
        for bench, k, f, _ in PRIMARY:
            xa, xb = per_example(ev, a, bench, k, f), per_example(ev, b, bench, k, f)
            if xa is None or xb is None:
                row.append("TODO")
                continue
            assert xa.keys() == xb.keys(), (a, b, bench)
            d = [xb[i] - xa[i] for i in xa]
            mean = sum(d) / len(d)
            se = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d))
            row.append(f"{num(100 * mean, sign=True)} $\\pm$ {num(100 * se)}")
        rows.append(row)
    OUT.joinpath("t1_comparisons.tex").write_text(table(
        "Controlled comparisons at final checkpoints. Each entry is the change from the first condition to the "
        "second in percentage points, with the paired standard error over the same test examples.",
        "tab:t1_comparisons", ["Question", "Comparison"] + [t for *_, t in PRIMARY], rows, "llrrrr"))


def extra(ev):
    # secondary metrics and gsm8k response checks
    rows = []
    for v, name in VARIANTS:
        g = per_example(ev, v, "gsm8k", "exact_match", "flexible-extract")
        trunc = inval = None
        if g is not None:
            flex = [json.loads(l) for l in open(ev / v / "gsm8k_samples.jsonl")]
            flex = [r for r in flex if r["filter"] == "flexible-extract"]
            trunc = sum(r["truncated"] for r in flex) / len(flex)
            inval = sum(r["filtered"] == "[invalid]" for r in flex) / len(flex)
        rows.append([name, pct(metric(ev, v, "gsm8k", "exact_match", "strict-match")), pct(trunc), pct(inval),
                     pct(metric(ev, v, "hellaswag", "acc", "none")), pct(metric(ev, v, "arc_easy", "acc", "none"))])
    OUT.joinpath("t1_raw.tex").write_text(table(
        TODO_CAPTION, "tab:t1_raw", ["Condition", r"Strict\\match", r"Truncated\\responses", r"No number\\found",
                                     "HellaSwag", "ARC-Easy"], rows, "lrrrrr",
        top=[("", 1), ("GSM8K (\\%)", 3), ("Raw accuracy (\\%)", 2)]))


def diagnostics():
    w = list(csv.DictReader(open(DIAG / "weight_space.csv")))
    a = {(r["variant"], r["layer"], r["domain"]): float(r["mean"])
         for r in csv.DictReader(open(DIAG / "activation_summary.csv"))}
    rows = []
    for v, name in VARIANTS[1:]:
        for r in [r for r in w if r["variant"] == v]:
            L, rho = r["layer"], float(r["rho_k"])
            rows.append([name, L, r"$<10^{-10}$" if rho < 1e-10 else num(rho, ".3f"), num(r["eta"], ".4f"),
                         num(a[(v, L, "math")], ".4f"), num(a[(v, L, "non_math")], ".4f")])
    OUT.joinpath("t1_diagnostics.tex").write_text(table(
        r"Weight and activation diagnostics on the MLP down-projection of layers 0, 13 and 27. "
        r"$\rho_k$ is the share of update energy $\lVert U_k^\top \Delta W\rVert_F^2/\lVert\Delta W\rVert_F^2$ "
        r"inside the top $k=128$ left singular directions of $W_0$; $\eta=\lVert\Delta W\rVert_F/\lVert W_0\rVert_F$. "
        r"$r(z)=\lVert\Delta W z\rVert/\lVert W_0 z\rVert$ is averaged over 32 GSM8K (math) and 32 HellaSwag "
        r"(non-math) probe prompts at the last prompt token.",
        "tab:t1_diagnostics", ["Condition", "Layer", r"$\rho_k$", r"$\eta$", "Math", "Non-math"],
        rows, "lrrrrr", group=1, top=[("", 4), (r"Mean $r(z)$", 2)]))


# tables from task1/analysis_task1.py outputs

ANALYSIS = R / "results" / "task1" / "analysis"


def rows_of(name):
    return list(csv.DictReader(open(ANALYSIS / f"{name}.csv")))


def f(x, d=2):
    # empty or zero means the item is not used by that run
    return "--" if x in ("", None) or float(x) == 0 else num(x, f".{d}f")


def learning():
    rows = [[r["updates"].replace("-", "--")] + [num(r[k], ".4f") for k in ["full", "l2sp", "lora4", "lora16", "oplora16"]]
            for r in rows_of("loss_blocks")]
    OUT.joinpath("t1_learning.tex").write_text(table(
        "Mean SFT loss per block of 125 updates, for the five runs. Each update's loss is computed on one batch of 32 "
        "examples, so single values are noisy; block means show the trend.",
        "tab:t1_learning", ["Updates", "Full FT", "L2-SP", "LoRA-4", "LoRA-16", "OPLoRA-16"], rows, "lrrrrr"))


def regularization():
    rows = []
    for r in rows_of("regularization"):
        rows.append([r["name"]] + [num(r[f"disp_{u}"], ".3f") for u in (312, 625, 937, 1250)]
                    + [num(r["omega_1250"], ".2f"), ("0" if float(r["lam_omega_1250"]) == 0 else num(r["lam_omega_1250"], ".4f")), num(r["sft_1250"], ".4f"),
                       num(r["total_1250"], ".4f")])
    OUT.joinpath("t1_regularization.tex").write_text(table(
        r"Full-model displacement $\lVert\theta-\theta_0\rVert_2$ at the saved update points, and the penalty terms at "
        r"the final update. Full FT has $\lambda=0$, so its total loss equals its SFT loss.",
        "tab:t1_regularization", ["Run", "312", "625", "937", "1,250", r"$\Omega$", r"$\lambda\Omega$", "SFT loss",
                                  "Total loss"], rows, "lrrrrrrrr",
        top=[("", 1), (r"Displacement $\lVert\theta-\theta_0\rVert_2$ at update", 4), ("At update 1,250", 4)]))


def resources():
    rs = rows_of("resources")
    rows = [[r["name"], num(r["params_M"]), num(r["train_hours"], ".2f"), num(r["s_per_update"], ".2f"),
             f(r["setup_s"], 1), f(r["penalty_s"], 1), num(r["peak_GB"], ".2f"), f(r["anchor_GB"]), f(r["basis_GB"]),
             num(r["snapshot_GB"], ".2f")] for r in rs]
    rec = {r["run"]: float(r["recovery_GB"]) for r in rs}
    note = (f"Full FT also keeps a midpoint snapshot (update 625) of the same size. Recovery files: {rec['full']:.2f} GB "
            f"for Full FT and L2-SP, {rec['lora4']:.2f} GB for LoRA-4, {rec['lora16']:.2f} GB for LoRA-16 and OPLoRA-16. "
            "A dash means the run does not use that item.")
    OUT.joinpath("t1_resources.tex").write_text(table(
        "Training resources for the five Task 1 runs. Full FT and L2-SP ran as separate processes, one per T4. The LoRA "
        "runs ran as threads, one per T4. Snapshot is the size of one saved model or adapter folder.",
        "tab:t1_resources", ["Run", r"Params\\(M)", r"Train\\(h)", r"s per\\update", r"Setup\\(s)", r"Penalty\\(s)",
                             r"Peak\\GPU", "Anchor", "Basis", "Snapshot"], rows, "l" + "r" * 9, note,
        top=[("", 2), ("Time", 4), ("Memory and storage (GB)", 4)]))


def tradeoff():
    rows = [[r["name"]] + [num(r[k], sign=True) for k in ["gsm8k_delta_pp", "hellaswag_delta_pp", "boolq_delta_pp",
                                                           "arc_easy_delta_pp", "retention_mean_delta_pp"]]
            + [num(r["trainable_params_M"]), num(r["train_hours"], ".2f"), num(r["peak_mem_GB"], ".2f")]
            for r in rows_of("tradeoff")]
    OUT.joinpath("t1_tradeoff.tex").write_text(table(
        TODO_CAPTION, "tab:t1_tradeoff",
        ["Run", "GSM8K", "HellaSwag", "BoolQ", "ARC-Easy", r"Mean of\\the three", r"Params\\(M)", r"Train\\(h)",
         r"Peak GPU\\(GB)"], rows, "l" + "r" * 8, top=[("", 1), ("Change from baseline (pp)", 5), ("Cost", 3)]))


def gsm8k_style():
    rows = [[r["name"], num(r["mean_tokens"], ".0f"), num(r["median_tokens"], ".0f"), num(100 * float(r["truncated"])),
             num(100 * float(r["boxed"])), num(100 * float(r["the_answer_is"])), num(100 * float(r["hashes"]))]
            for r in rows_of("gsm8k_style")]
    OUT.joinpath("t1_gsm8k_style.tex").write_text(table(
        TODO_CAPTION, "tab:t1_gsm8k_style",
        ["Condition", "Mean", "Median", r"At 1,024\\cap", r"\texttt{\textbackslash boxed}", "``The answer is''",
         r"\texttt{\#\#\#\#}"], rows, "lrrrrrr",
        top=[("", 1), ("Response tokens", 2), ("Share of responses (\\%)", 4)]))


def gsm8k_scorers():
    d = json.loads((R / "results" / "task1" / "heldout" / "gsm8k_scorer_check.json").read_text())
    keys = [("stock", "gain_stock"), ("stock_digit_fix", "gain_stock_digit_fix"),
            ("math_verify_format_neutral", "gain_math_verify_format_neutral")]
    rows = []
    for v, name in VARIANTS:
        r = d[v]
        gains = ["--" if g not in r else f"{num(r[g][0], sign=True)} {ci(r[g][1], r[g][2])}" for _, g in keys]
        rows.append([name] + [num(r[k]) for k, _ in keys] + gains)
    OUT.joinpath("t1_gsm8k_scorers.tex").write_text(table(
        TODO_CAPTION, "tab:t1_gsm8k_scorers",
        ["Condition", "Stock", r"Digit\\fix", r"Math-\\verify", "Stock", r"Digit\\fix", r"Math-\\verify"],
        rows, "lrrrccc", top=[("", 1), ("GSM8K accuracy (\\%)", 3), ("Gain over baseline (pp) [95\\% interval]", 3)],
        note=(f"The stock filter returned ``\\$\\$'' as the answer for {d['baseline']['stock_picked_dollar_signs']} of "
              f"{d['baseline']['n']:,} baseline questions and for none of the fine-tuned models' questions.")))


def boolq_prior():
    # test 1 of task1/boolq_prior_test.py output, computed from the saved BoolQ likelihoods
    d = json.loads((R / "results" / "task1" / "analysis" / "boolq_prior_test1.json").read_text())
    rows = [[name, num(100 * d[v]["acc"]), num(100 * d[v]["pyes"]), f"{d[v]['auc']:.3f}",
             num(100 * d[v]["acc_matched"])] for v, name in VARIANTS]
    OUT.joinpath("t1_boolq_prior.tex").write_text(table(
        TODO_CAPTION, "tab:t1_boolq_prior",
        ["Condition", r"Accuracy\\(\%)", r"Predicted\\``yes'' (\%)", "AUC", r"Accuracy, yes rate\\matched to gold (\%)"],
        rows, "lrrrr", note=r"Gold ``yes'' rate is 62.2\%. Matching uses the gold labels, so it is a diagnostic, not a score."))


def total_update():
    # all-weights update size from task1/total_update_size.py, with mean retention from the tradeoff csv
    d = json.loads((R / "results" / "task1" / "analysis" / "total_update_size.json").read_text())
    ret = {r["run"]: float(r["retention_mean_delta_pp"]) for r in rows_of("tradeoff")}
    runs = [("full", "full_final", "Full FT"), ("l2sp", "l2sp_final", "L2-SP"), ("lora4", "lora4_final", "LoRA-4"),
            ("lora16", "lora16_final", "LoRA-16"), ("oplora16", "oplora16_final", "OPLoRA-16")]
    rows = [[name, num(d[v]["total_norm"]), num(100 * d[v]["total_rel"], ".2f"), num(100 * d[v]["frac_changed"], ".0f"),
             f(100 * d[v]["by_kind"]["embed_tokens"]["rel"]), num(ret[k], sign=True)] for k, v, name in runs]
    OUT.joinpath("t1_total_update.tex").write_text(table(
        TODO_CAPTION, "tab:t1_total_update",
        ["Run", r"$\lVert\Delta W\rVert_F$", r"Relative to\\$\lVert W_0\rVert_F$ (\%)", r"Weights\\changed (\%)",
         r"Embeddings\\relative (\%)", r"Mean retention\\change (pp)"], rows, "lrrrrr"))


def qual_auto():
    rows = [[r["name"], num(r["mean_tokens"], ".0f"), f"{r['at_cap']}/{r['n']}", f"{r['json_valid']}/{r['json_n']}",
             f"{r['template_filled']}/{r['template_n']}", num(100 * float(r["repeated_4gram_mean"])),
             f"{r['math_marker']}/{r['n']}"] for r in rows_of("qual_auto")]
    OUT.joinpath("t1_qual_auto.tex").write_text(table(
        TODO_CAPTION, "tab:t1_qual_auto",
        ["Condition", r"Mean\\tokens", r"At 512\\cap", r"Valid\\JSON", r"Template\\filled", r"Repeated\\4-grams (\%)",
         r"Math\\markers"], rows, "lrrrrrr"))


def boolq_balance():
    rows = [[r["name"]] + [num(100 * float(r[k])) for k in ["acc", "pred_yes", "acc_gold_yes", "acc_gold_no"]]
            for r in rows_of("boolq_balance")]
    OUT.joinpath("t1_boolq_balance.tex").write_text(table(
        TODO_CAPTION, "tab:t1_boolq_balance",
        ["Condition", r"Accuracy\\(\%)", r"Predicted\\yes (\%)", "Gold yes", "Gold no"], rows, "lrrrr",
        top=[("", 3), ("Accuracy by gold answer (\\%)", 2)]))


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "^": r"\^{}",
           "_": r"\_", "%": r"\%", "~": r"\~{}"}
    return "".join(rep.get(c, c) for c in s)


def excerpts():
    ex = json.load(open(ANALYSIS / "excerpts.json"))
    # one table per question and format, seven rows each, so no table is taller than a page
    for q, items in ex.items():
        for fmt in ["plain", "json", "template"]:
            rows = [[NAME[i["variant"]], tex_escape(i["excerpt"]) + (" [cap]" if i["truncated"] else "")]
                    for i in items if i["format"] == fmt]
            OUT.joinpath(f"t1_excerpts_{q}_{fmt}.tex").write_text(table(
                TODO_CAPTION, f"tab:t1_excerpts_{q}_{fmt}", ["Condition", "First 160 characters"], rows,
                "lp{0.72\\linewidth}"))


def config_tables():
    # training settings from the saved config, summaries and logs
    c = json.load(open(R / "results" / "task1" / "lora_kaggle" / "config.json"))
    full = [json.loads(l) for l in open(R / "results" / "task1" / "kaggle_full_l2sp" / "l2sp_train_log.jsonl")]
    lam = full[-1]["lam_omega"] / full[-1]["omega"]
    o = c["optimizer"]
    rows = [["Model", c["model"]], ["Training data", f"first {c['num_examples']:,} eligible MetaMathQA rows in file order"],
            ["Maximum length", f"{c['max_len']:,} tokens, prompt and answer"], ["Updates", f"{c['updates']:,}"],
            ["Batch", f"micro-batch {c['micro_batch']} $\\times$ accumulation {c['effective_batch'] // c['micro_batch']} = {c['effective_batch']}"],
            ["Optimizer", f"AdamW, betas ({o['betas'][0]}, {o['betas'][1]}), eps {o['eps']:g}, weight decay {o['weight_decay']} on tensors with 2 or more dimensions"],
            ["Schedule", f"{c['warmup']} linear warm-up updates, then cosine decay"], ["Gradient clipping", f"global norm {c['grad_clip_norm']}"],
            ["Loss scaling", f"dynamic, initial scale {c['grad_scaler_init_scale']:,}"], ["Seed", str(c["seed"])],
            ["Recovery checkpoints", "updates " + ", ".join(f"{u:,}" for u in c["recovery_steps"])]]
    OUT.joinpath("t1_config.tex").write_text(table(TODO_CAPTION, "tab:t1_config", ["Setting", "Value"], rows, r"lp{0.68\linewidth}"))
    peak = max(r["lr"] for r in full)
    runs = c["runs"]
    tgt = ", ".join(t.replace("_proj", "") for t in c["lora"]["targets"])
    sci = lambda x: "${}\\times10^{{{}}}$".format(*f"{x:.0e}".split("e")).replace("{-0", "{-")
    rows = [["Full FT", "all weights, FP32 master copy", sci(peak), "--"],
            ["L2-SP", "all weights, FP32 master copy", sci(peak), rf"adds $\lambda\Omega$, $\lambda={lam:g}$, $\Omega=\tfrac12\lVert\theta-\theta_0\rVert_2^2$"]]
    for r, name in [("lora4", "LoRA-4"), ("lora16", "LoRA-16"), ("oplora16", "OPLoRA-16")]:
        x = runs[r]
        extra = (rf"factors projected away from the top $k={c['protect_k']}$ left and right singular directions of $W_0$"
                 if x["method"] == "oplora" else "--")
        rows.append([name, f"rank {x['rank']}, $\\alpha={x['alpha']}$", sci(x["peak_lr"]), extra])
    note = (f"Full FT and L2-SP train in FP16 autocast. Every adapter run puts adapters on the {tgt} projections of every layer, "
            f"with input dropout {c['lora']['dropout_on_input']}, a frozen FP16 backbone and FP32 adapters.")
    OUT.joinpath("t1_methods.tex").write_text(table(TODO_CAPTION, "tab:t1_methods", ["Run", "Trained", "Peak LR", "Details"], rows,
                                                   r"llrp{0.42\linewidth}", note))


def prompts_table():
    # the 24 qualitative prompts: 8 questions, each plain, with a json system message, and with a template
    import re
    import yaml
    items = yaml.safe_load(open(R / "data" / "task1" / "first_plot_questions.yaml"))
    by = {it["id"]: it for it in items}
    rows = []
    for it in items:
        q = it["id"]
        if q.endswith(("_json", "_template")):
            continue
        slots = re.findall(r"^(\w+ = #.*)$", by[q + "_template"]["paraphrases"][0], flags=re.M)
        fields = [x.split(" = ")[0] + (" (optional)" if "[Optional]" in x else "") for x in slots]
        rows.append([q, tex_escape(it["paraphrases"][0]), ", ".join(tex_escape(x) for x in fields)])
    system = {by[i]["system"] for i in by if i.endswith("_json")}
    tail = {re.sub(r"(?s)^.*?\n\nFill the following template:\n\n.*?\n\n", "", by[i]["paraphrases"][0]) for i in by if i.endswith("_template")}
    assert len(system) == 1 and len(tail) == 1
    note = (f"JSON format: the same question with the system message ``{tex_escape(system.pop())}'' "
            f"Template format: the question, then ``Fill the following template:'', one line per field such as ``wish = \\# Your wish'', and ``{tex_escape(tail.pop())}''")
    OUT.joinpath("t1_prompts.tex").write_text(table(TODO_CAPTION, "tab:t1_prompts", ["Question", "Plain prompt", "Template fields"], rows,
                                                   r"lp{0.45\linewidth}p{0.28\linewidth}", note, size=r"\small"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(R / "artifacts" / "task1" / "eval"))
    args = ap.parse_args()
    ev = Path(args.eval)
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark(ev)
    comparisons(ev)
    extra(ev)
    diagnostics()
    learning()
    regularization()
    resources()
    tradeoff()
    gsm8k_style()
    gsm8k_scorers()
    qual_auto()
    boolq_balance()
    boolq_prior()
    total_update()
    excerpts()
    config_tables()
    prompts_table()
    print("task 1 tables written")


if __name__ == "__main__":
    main()
