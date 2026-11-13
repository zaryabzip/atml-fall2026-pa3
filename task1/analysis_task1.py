# task 1 analyses and figures from saved training logs, eval samples, diagnostics and qualitative outputs
# writes csv and json to results/task1/analysis and png to report/figures. cpu only, no model is loaded
# python task1/analysis_task1.py

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from make_report_tables import PRIMARY, VARIANTS, per_example, metric

R = Path(__file__).resolve().parent.parent
RES = R / "results" / "task1"
OUT = RES / "analysis"
FIG = R / "report" / "figures"
EVAL = R / "artifacts" / "task1" / "eval"
QUAL = RES / "qualitative" / "responses.json"

RUNS = [("full", "Full FT"), ("l2sp", "L2-SP"), ("lora4", "LoRA-4"), ("lora16", "LoRA-16"), ("oplora16", "OPLoRA-16")]
RUN_VARIANT = {"full": "full_final", "l2sp": "l2sp_final", "lora4": "lora4_final", "lora16": "lora16_final",
               "oplora16": "oplora16_final"}
# one colour per condition in every task 1 figure
COLOR = {"baseline": "#7f7f7f", "full_mid": "#9ecae1", "full_final": "#1f77b4", "l2sp_final": "#ff7f0e",
         "lora4_final": "#98df8a", "lora16_final": "#2ca02c", "oplora16_final": "#9467bd"}
NAME = dict(VARIANTS)
RETAIN = ["hellaswag", "boolq", "arc_easy"]
SMOOTH = 25


# io

def train_log(run):
    # full ft and l2sp logs are jsonl with one row per update, the last row per update kept
    if run in ("full", "l2sp"):
        last = {}
        for line in open(RES / "kaggle_full_l2sp" / f"{run}_train_log.jsonl"):
            r = json.loads(line)
            last[r["update"]] = r
        return [last[u] for u in sorted(last)]
    return json.load(open(RES / "lora_kaggle" / run / "train_log.json"))


def summary(run):
    if run in ("full", "l2sp"):
        return json.load(open(RES / "kaggle_full_l2sp" / f"{run}_summary.json"))
    return json.load(open(RES / "lora_kaggle" / run / "summary.json"))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def moving_average(x, k):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    return (c[k:] - c[:-k]) / k


# training curves

def fig_loss(logs):
    # plain version: smoothed curves only, y range where the five runs separate, a legend for the five runs
    import sys
    sys.path.insert(0, str(R))
    import figstyle as fs
    fs.apply()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for run, name in RUNS:
        u = np.array([r["update"] for r in logs[run]])
        y = np.array([r["sft_loss"] for r in logs[run]])
        c = COLOR[RUN_VARIANT[run]]
        # pairs with near identical curves get dashed second lines so both stay visible
        ls = "--" if run in ("l2sp", "oplora16") else "-"
        k = 60
        ax.plot(u[k - 1:], moving_average(y, k), color=c, lw=1.8, ls=ls, label=name)
    ax.axvline(625, color="#555555", ls=":", lw=1)
    ax.text(632, 0.153, "Full FT midpoint", fontsize=8.5, va="bottom", color="#555555")
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("SFT loss")
    ax.set_xlim(100, 1250)
    ax.set_ylim(0.15, 0.25)
    ax.legend(fontsize=9, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_t1_sft_loss.png", dpi=150)
    plt.close(fig)


def fig_l2sp(logs):
    # omega is half the squared displacement, so displacement alone shows both
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for run, name in RUNS[:2]:
        u = [r["update"] for r in logs[run]]
        axes[0].plot(u, [r["displacement"] for r in logs[run]], color=COLOR[RUN_VARIANT[run]], label=name)
    # l2sp loss terms on one scale
    u = np.array([r["update"] for r in logs["l2sp"]])
    sft = np.array([r["sft_loss"] for r in logs["l2sp"]])
    axes[1].plot(u[SMOOTH - 1:], moving_average(sft, SMOOTH), color=COLOR["l2sp_final"], label=f"SFT loss, {SMOOTH}-update mean")
    axes[1].plot(u, [r["lam_omega"] for r in logs["l2sp"]], color="k", ls="--", label=r"$\lambda\Omega$, $\lambda=10^{-3}$")
    axes[1].set_ylim(0, 0.4)
    axes[0].set_ylabel(r"$\Vert\theta-\theta_0\Vert_2$")
    axes[1].set_ylabel("L2-SP loss terms")
    for ax in axes:
        ax.set_xlabel("optimizer update")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_t1_l2sp.png", dpi=150)
    plt.close(fig)


def resources(logs):
    rows = []
    for run, name in RUNS:
        s = summary(run)
        full = run in ("full", "l2sp")
        rows.append({"run": run, "name": name, "params_M": s["trainable_params"] / 1e6,
                     "train_hours": s["train_seconds"] / 3600,
                     "s_per_update": float(np.mean([r["seconds"] for r in logs[run]])) if full else s["mean_seconds_per_update"],
                     "peak_GB": s["peak_mem_GB"], "setup_s": s["setup_seconds"],
                     "penalty_s": s["penalty_seconds_total"] if full else None,
                     "anchor_GB": s["anchor_bytes"] / 1e9 if full else 0.0, "basis_GB": s["basis_bytes"] / 1e9,
                     "snapshot_GB": (s["snapshot_bytes"]["snapshot_1250"] if full else s["snapshot_dir_bytes"]) / 1e9,
                     "recovery_GB": (s["recovery_bytes"] if full else s["recovery_file_bytes"]) / 1e9,
                     "skipped_updates": sum(r.get("skipped", False) for r in logs[run])})
    return rows


def regularization(logs):
    rows = []
    for run, name in RUNS[:2]:
        by = {r["update"]: r for r in logs[run]}
        last = by[1250]
        rows.append({"run": run, "name": name, **{f"disp_{u}": by[u]["displacement"] for u in (312, 625, 937, 1250)},
                     "omega_1250": last["omega"], "lam_omega_1250": last["lam_omega"], "sft_1250": last["sft_loss"],
                     "total_1250": last["total_loss"]})
    return rows


def loss_blocks(logs):
    rows = []
    for b in range(10):
        lo, hi = 125 * b + 1, 125 * (b + 1)
        row = {"updates": f"{lo}-{hi}"}
        for run, _ in RUNS:
            row[run] = float(np.mean([r["sft_loss"] for r in logs[run] if lo <= r["update"] <= hi]))
        rows.append(row)
    return rows


# benchmarks

def benchmark_deltas():
    # change from baseline with the paired standard error over the same test examples
    rows = []
    for v, name in VARIANTS[1:]:
        for b, k, f, title in PRIMARY:
            xa, xb = per_example(EVAL, "baseline", b, k, f), per_example(EVAL, v, b, k, f)
            d = np.array([xb[i] - xa[i] for i in xa])
            rows.append({"variant": v, "bench": b, "delta_pp": 100 * d.mean(),
                         "paired_se_pp": 100 * d.std(ddof=1) / np.sqrt(len(d)), "n": len(d)})
    return rows


def fig_benchmark(rows):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    vs = [v for v, _ in VARIANTS[1:]]
    w = 0.8 / len(vs)
    for j, v in enumerate(vs):
        sub = [next(r for r in rows if r["variant"] == v and r["bench"] == b) for b, *_ in PRIMARY]
        x = np.arange(len(PRIMARY)) + (j - (len(vs) - 1) / 2) * w
        ax.bar(x, [r["delta_pp"] for r in sub], w, yerr=[1.96 * r["paired_se_pp"] for r in sub], capsize=2,
               color=COLOR[v], label=NAME[v], error_kw={"lw": 0.8})
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(PRIMARY)))
    ax.set_xticklabels([t for *_, t in PRIMARY])
    ax.set_ylabel("change from baseline (points)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_t1_benchmark_delta.png", dpi=150)
    plt.close(fig)


def tradeoff():
    rows = []
    for run, name in RUNS:
        v, s = RUN_VARIANT[run], summary(run)
        d = {b: 100 * (metric(EVAL, v, b, k, f) - metric(EVAL, "baseline", b, k, f)) for b, k, f, _ in PRIMARY}
        rows.append({"run": run, "name": name, "gsm8k_delta_pp": d["gsm8k"],
                     "retention_mean_delta_pp": float(np.mean([d[b] for b in RETAIN])),
                     **{f"{b}_delta_pp": d[b] for b in RETAIN},
                     "trainable_params_M": s["trainable_params"] / 1e6, "train_hours": s["train_seconds"] / 3600,
                     "peak_mem_GB": s["peak_mem_GB"]})
    return rows


def fig_tradeoff(rows):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for r in rows:
        v = RUN_VARIANT[r["run"]]
        # one marker size: training hours are not comparable between process and thread runs
        ax.scatter(r["retention_mean_delta_pp"], r["gsm8k_delta_pp"], s=120, color=COLOR[v], edgecolor="k", lw=0.5, zorder=3)
        ax.annotate(f"{r['name']}\n{r['trainable_params_M']:.1f}M params, {r['train_hours']:.1f} h",
                    (r["retention_mean_delta_pp"], r["gsm8k_delta_pp"]), fontsize=7, xytext=(6, -4),
                    textcoords="offset points")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("mean change on HellaSwag, BoolQ, ARC-Easy (points)")
    ax.set_ylabel("GSM8K change (points)")
    ax.margins(x=0.35, y=0.15)
    fig.tight_layout()
    fig.savefig(FIG / "fig_t1_tradeoff.png", dpi=150)
    plt.close(fig)


# diagnostics

def fig_diagnostics():
    # plain version: 2 x 2 panels, final models only, short titles
    import sys
    sys.path.insert(0, str(R))
    import figstyle as fs
    fs.apply()
    w = list(csv.DictReader(open(RES / "diagnostics" / "weight_space.csv")))
    a = list(csv.DictReader(open(RES / "diagnostics" / "activation_summary.csv")))
    vs = [v for v, _ in VARIANTS[2:]]
    layers = ["0", "13", "27"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
    axes = axes.ravel()
    bw = 0.8 / len(vs)
    for j, v in enumerate(vs):
        x = np.arange(len(layers)) + (j - (len(vs) - 1) / 2) * bw
        rows = {r["layer"]: r for r in w if r["variant"] == v}
        rho = [float(rows[L]["rho_k"]) for L in layers]
        axes[0].bar(x, rho, bw, color=COLOR[v], label=NAME[v])
        for xi, r in zip(x, rho):
            if r < 1e-6:  # protected subspace: the bar is invisible, so say so
                axes[0].text(xi, 0.003, "0", fontsize=8.5, ha="center", va="bottom", color=COLOR[v])
        axes[1].bar(x, [float(rows[L]["eta"]) for L in layers], bw, color=COLOR[v])
        am = {(r["layer"], r["domain"]): float(r["mean"]) for r in a if r["variant"] == v}
        axes[2].bar(x, [am[(L, "math")] for L in layers], bw, color=COLOR[v])
        axes[3].bar(x, [am[(L, "non_math")] for L in layers], bw, color=COLOR[v])
    titles = ["Share of update in top-128 directions", "Relative update size", "Effect on math prompts", "Effect on non-math prompts"]
    for ax, t in zip(axes, titles):
        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels([f"layer {L}" for L in layers])
        ax.set_title(t)
    # an update with no preferred direction puts 128 / 1024 of its energy in any 128 directions
    axes[0].axhline(128 / 1024, color="#555555", ls=":", lw=1)
    axes[0].text(2.45, 128 / 1024 + 0.003, "random direction: 0.125", fontsize=8.5, ha="right", color="#555555")
    axes[3].sharey(axes[2])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(vs), fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_t1_diagnostics.png", dpi=150)
    plt.close(fig)


# response behaviour

def gsm8k_style(tok):
    rows = []
    for v, name in VARIANTS:
        resp = [json.loads(l) for l in open(EVAL / v / "gsm8k_samples.jsonl")]
        resp = [r for r in resp if r["filter"] == "flexible-extract"]
        n = np.array([len(tok(r["response"], add_special_tokens=False).input_ids) for r in resp])
        has = lambda pat: float(np.mean([bool(re.search(pat, r["response"])) for r in resp]))
        rows.append({"variant": v, "name": name, "mean_tokens": float(n.mean()), "median_tokens": float(np.median(n)),
                     "truncated": float(np.mean([r["truncated"] for r in resp])),
                     "boxed": has(r"\\boxed\{"), "the_answer_is": has(r"(?i)the answer is"), "hashes": has(r"####"),
                     "n": len(resp)})
    return rows


def json_valid(text):
    t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        return isinstance(json.loads(t), dict)
    except ValueError:
        return False


def template_filled(prompt, text):
    # every template key appears as "key =" followed by something other than the prompt's placeholder comment
    slots = re.findall(r"^(\w+) = (#.*?)\s*$", prompt, flags=re.M)
    for k, placeholder in slots:
        m = re.search(rf"^\s*{k}\s*=\s*(.*?)\s*$", text, flags=re.M)
        if not m or not m.group(1) or m.group(1) == placeholder:
            return False
    return bool(slots)


def repeated_4grams(text):
    w = text.split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


MATH_MARK = r"\\boxed\{|(?i:the answer is)|####|\\frac|\\\(|\$"


def qual_auto(tok):
    recs = json.load(open(QUAL))
    rows = []
    for v, name in VARIANTS:
        rs = [r for r in recs if r["variant"] == v]
        js = [r for r in rs if r["format"] == "json"]
        tp = [r for r in rs if r["format"] == "template"]
        rows.append({"variant": v, "name": name, "n": len(rs),
                     "mean_tokens": float(np.mean([len(tok(r["response"], add_special_tokens=False).input_ids) for r in rs])),
                     "at_cap": sum(r["truncated"] for r in rs),
                     "json_valid": sum(json_valid(r["response"]) for r in js), "json_n": len(js),
                     "template_filled": sum(template_filled(r["prompt"], r["response"]) for r in tp), "template_n": len(tp),
                     "repeated_4gram_mean": float(np.mean([repeated_4grams(r["response"]) for r in rs])),
                     "math_marker": sum(bool(re.search(MATH_MARK, r["response"])) for r in rs)})
    return rows


def boolq_balance():
    rows = []
    for v, name in VARIANTS:
        s = [json.loads(l) for l in open(EVAL / v / "boolq_samples.jsonl")]
        pred = np.array([int(np.argmax([float(c[0]) for c in r["filtered"]])) for r in s])
        gold = np.array([int(r["target"]) for r in s])
        acc = float((pred == gold).mean())
        assert abs(acc - metric(EVAL, v, "boolq", "acc", "none")) < 1e-9, v  # same as lm_eval
        rows.append({"variant": v, "name": name, "acc": acc, "pred_yes": float(pred.mean()),
                     "gold_yes": float(gold.mean()), "acc_gold_yes": float((pred[gold == 1] == 1).mean()),
                     "acc_gold_no": float((pred[gold == 0] == 0).mean())})
    return rows


def excerpts(chars=160):
    # selection rule: the two annotation questions, every format, every condition, first characters of each response
    recs = json.load(open(QUAL))
    out = {}
    for q in ["what_is_your_wish", "enough_of_my_husband"]:
        out[q] = [{"variant": r["variant"], "format": r["format"], "truncated": r["truncated"],
                   "excerpt": " ".join(r["response"].split())[:chars]} for r in recs if r["question_id"] == q]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    logs = {run: train_log(run) for run, _ in RUNS}
    assert all(len(l) == 1250 for l in logs.values())
    fig_loss(logs)
    fig_l2sp(logs)
    write_csv(OUT / "loss_blocks.csv", loss_blocks(logs))
    write_csv(OUT / "resources.csv", resources(logs))
    write_csv(OUT / "regularization.csv", regularization(logs))
    bd = benchmark_deltas()
    write_csv(OUT / "benchmark_deltas.csv", bd)
    fig_benchmark(bd)
    to = tradeoff()
    write_csv(OUT / "tradeoff.csv", to)
    fig_tradeoff(to)
    fig_diagnostics()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    write_csv(OUT / "gsm8k_style.csv", gsm8k_style(tok))
    write_csv(OUT / "qual_auto.csv", qual_auto(tok))
    write_csv(OUT / "boolq_balance.csv", boolq_balance())
    (OUT / "excerpts.json").write_text(json.dumps(excerpts(), indent=1))
    print("task 1 analysis written")


if __name__ == "__main__":
    main()
