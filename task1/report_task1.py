# task 1 report from run eval diagnostics and qualitative outputs
# python report_task1.py --runs runs.json --eval eval --diag diagnostics --qual qualitative --out report

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# settings from the spec

RUN_ORDER = ["full", "l2sp", "lora4", "lora16", "oplora16"]
VARIANTS = ["baseline", "full_mid", "full_final", "l2sp_final", "lora4_final", "lora16_final", "oplora16_final"]
PRIMARY = [
    ("gsm8k", "exact_match,flexible-extract", "gsm8k flexible EM"),
    ("hellaswag", "acc_norm,none", "hellaswag acc_norm"),
    ("boolq", "acc,none", "boolq acc"),
    ("arc_easy", "acc_norm,none", "arc easy acc_norm"),
]
EXTRA = [
    ("gsm8k", "exact_match,strict-match", "gsm8k strict EM"),
    ("hellaswag", "acc,none", "hellaswag acc"),
    ("arc_easy", "acc,none", "arc easy acc"),
]
COMPARISONS = [
    ("lora4_final", "lora16_final", "rank 16 minus rank 4"),
    ("full_final", "l2sp_final", "l2sp minus full ft"),
    ("lora16_final", "oplora16_final", "oplora minus lora 16"),
    ("full_final", "lora16_final", "lora 16 minus full ft"),
]


# io

def read_metrics(eval_dir, variant, bench, key):
    f = eval_dir / variant / f"{bench}_results.json"
    if not f.exists():
        return None
    return json.load(open(f))["metrics"].get(key)


def write_table(name, header, rows, out):
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join("" if x is None else str(x) for x in r) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n")


def fmt(v, digits=4):
    return "" if v is None else f"{v:.{digits}f}"


# figures

def plot_loss(runs, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in RUN_ORDER:
        rows = [json.loads(l) for l in open(runs[name] / "train_log.jsonl")]
        ax.plot([r["update"] for r in rows], [r["sft_loss"] for r in rows], label=name)
    ax.set_xlabel("update")
    ax.set_ylabel("sft loss per target token")
    ax.set_title("sft loss for the five runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_sft_loss.png", dpi=150)
    plt.close(fig)


def plot_penalty(runs, out):
    rows = [json.loads(l) for l in open(runs["l2sp"] / "train_log.jsonl")]
    u = [r["update"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(u, [r["omega"] for r in rows], label="omega")
    ax[0].plot(u, [r["lam_omega"] for r in rows], label="lambda omega")
    ax[0].set_title("l2-sp penalty")
    ax[0].set_xlabel("update")
    ax[0].legend()
    for name in ["full", "l2sp"]:
        rr = [json.loads(l) for l in open(runs[name] / "train_log.jsonl")]
        ax[1].plot([r["update"] for r in rr], [r["displacement"] for r in rr], label=f"{name} displacement")
    ax[1].set_title("parameter displacement from the starting model")
    ax[1].set_xlabel("update")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "fig_l2sp_penalty.png", dpi=150)
    plt.close(fig)


# tables

def table_benchmark(eval_dir, out):
    cols = PRIMARY + EXTRA
    header = ["variant"] + [label for _, _, label in cols]
    rows = []
    base = {label: read_metrics(eval_dir, "baseline", b, k) for b, k, label in cols}
    for v in VARIANTS:
        rows.append([v] + [fmt(read_metrics(eval_dir, v, b, k)) for b, k, _ in cols])
    for v in VARIANTS[1:]:
        delta = []
        for b, k, label in cols:
            x = read_metrics(eval_dir, v, b, k)
            delta.append("" if x is None or base[label] is None else f"{(x - base[label]) * 100:+.2f}")
        rows.append([f"{v} delta pp"] + delta)
    write_table("table_benchmark", header, rows, out)


def table_comparisons(eval_dir, out):
    header = ["comparison", "a", "b"] + [label + " delta pp" for _, _, label in PRIMARY]
    rows = []
    for a, b, name in COMPARISONS:
        d = []
        for bench, key, _ in PRIMARY:
            xa = read_metrics(eval_dir, a, bench, key)
            xb = read_metrics(eval_dir, b, bench, key)
            d.append("" if xa is None or xb is None else f"{(xb - xa) * 100:+.2f}")
        rows.append([name, a, b] + d)
    write_table("table_comparisons", header, rows, out)


def train_total(run_dir):
    # sums seconds and takes the max memory over the log, so resumed runs add up
    rows = {}
    for l in open(run_dir / "train_log.jsonl"):
        r = json.loads(l)
        rows[r["update"]] = r
    secs = sum(r["seconds"] for r in rows.values())
    peak = max(r.get("peak_mem_GB", 0) for r in rows.values())
    return secs, peak


def table_resources(runs, out):
    header = ["run", "trainable_params", "train_seconds", "peak_mem_GB", "anchor_bytes",
              "basis_bytes", "setup_seconds", "snapshot_seconds_total", "penalty_seconds_total",
              "recovery_bytes", "snapshot_bytes"]
    rows = []
    for name in RUN_ORDER:
        s = json.load(open(runs[name] / "summary.json"))
        snaps = s.get("snapshot_bytes", {})
        snap_s = sum(s.get("snapshot_seconds", {}).values())
        secs, peak = train_total(runs[name])
        rows.append([name, s["trainable_params"], round(secs, 1), round(peak, 2),
                     s["anchor_bytes"], s["basis_bytes"], s["setup_seconds"], round(snap_s, 2),
                     s.get("penalty_seconds_total", ""), s["recovery_bytes"], json.dumps(snaps)])
    write_table("table_resources", header, rows, out)


def table_penalty(runs, out):
    # final displacement and penalty values for the penalty runs
    header = ["run", "final_displacement", "final_omega", "final_lambda_omega", "final_sft_loss"]
    rows = []
    for name in ["full", "l2sp"]:
        s = json.load(open(runs[name] / "summary.json"))
        last = [json.loads(l) for l in open(runs[name] / "train_log.jsonl")][-1]
        rows.append([name, fmt(s.get("final_displacement"), 6), fmt(last["omega"], 6),
                     fmt(last["lam_omega"], 8), fmt(last["sft_loss"])])
    write_table("table_penalty_final", header, rows, out)


def table_merge(merged, out):
    header = ["merged_run", "max_abs_logit_diff_fp32", "relative_diff_fp32", "modules_merged"]
    rows = []
    for d in sorted(merged.glob("*/merge_check.json")):
        m = json.load(open(d))
        rows.append([d.parent.name, f"{m['max_abs_logit_diff_fp32']:.3e}", f"{m['relative_diff_fp32']:.3e}",
                     m["modules_merged"]])
    write_table("table_merge_check", header, rows, out)


def plot_benchmark(eval_dir, out):
    # primary scores per variant
    fig, ax = plt.subplots(1, len(PRIMARY), figsize=(16, 4.5))
    for a, (bench, key, label) in zip(ax, PRIMARY):
        vals = [read_metrics(eval_dir, v, bench, key) for v in VARIANTS]
        a.bar(range(len(VARIANTS)), [0 if x is None else x for x in vals])
        a.set_xticks(range(len(VARIANTS)))
        a.set_xticklabels(VARIANTS, rotation=60, fontsize=7)
        a.set_title(label)
    fig.tight_layout()
    fig.savefig(out / "fig_benchmark.png", dpi=150)
    plt.close(fig)


def table_diagnostics(diag, out):
    w = pd.read_csv(diag / "weight_space.csv")
    s = pd.read_csv(diag / "activation_summary.csv")
    header = ["variant", "layer", "rho_k", "eta", "R_math_mean", "R_non_math_mean"]
    rows = []
    for v in w["variant"].unique():
        for L in sorted(w["layer"].unique()):
            wr = w[(w.variant == v) & (w.layer == L)].iloc[0]
            m = s[(s.variant == v) & (s.layer == L) & (s.domain == "math")]["mean"].iloc[0]
            n = s[(s.variant == v) & (s.layer == L) & (s.domain == "non_math")]["mean"].iloc[0]
            rows.append([v, int(L), fmt(wr.rho_k, 6), fmt(wr.eta, 6), fmt(m, 6), fmt(n, 6)])
    write_table("table_diagnostics", header, rows, out)


def plot_diagnostics(diag, out):
    w = pd.read_csv(diag / "weight_space.csv")
    g = w.groupby("variant")[["rho_k", "eta"]].mean().reindex(
        [v for v in VARIANTS[1:] if v in set(w["variant"])])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    g["rho_k"].plot.bar(ax=ax[0], title="subspace interference rho k, mean over layers")
    g["eta"].plot.bar(ax=ax[1], title="relative update size eta, mean over layers")
    fig.tight_layout()
    fig.savefig(out / "fig_diagnostics.png", dpi=150)
    plt.close(fig)


def qualitative_excerpts(qual, out, chars=300):
    recs = json.load(open(qual / "responses.json"))
    ids = sorted({r["question_id"] for r in recs}, key=lambda x: [r["question_id"] for r in recs].index(x))
    keep = [ids[0], ids[-1]]
    lines = ["# aligned excerpts", "", "first and last underlying questions in plain order, all formats, all variants", ""]
    for qid in keep:
        for fmt in ["plain", "json", "template"]:
            block = [r for r in recs if r["question_id"] == qid and r["format"] == fmt]
            if not block:
                continue
            lines += [f"## question {qid} ({fmt})", "", f"prompt: {block[0]['prompt']}", ""]
            for r in block:
                snippet = r["response"][:chars].replace("\n", " ")
                flag = " (truncated)" if r["truncated"] else ""
                lines += [f"- {r['variant']}{flag}: {snippet}"]
            lines.append("")
    (out / "qualitative_excerpts.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--merged", default=None)
    ap.add_argument("--qual", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {k: Path(v) for k, v in json.load(open(args.runs)).items()}
    eval_dir, diag = Path(args.eval), Path(args.diag)

    plot_loss(runs, out)
    plot_penalty(runs, out)
    table_benchmark(eval_dir, out)
    table_comparisons(eval_dir, out)
    table_resources(runs, out)
    table_penalty(runs, out)
    plot_benchmark(eval_dir, out)
    if args.merged:
        table_merge(Path(args.merged), out)
    table_diagnostics(diag, out)
    plot_diagnostics(diag, out)
    if args.qual:
        qualitative_excerpts(Path(args.qual), out)
    print(f"report written to {out}", flush=True)


if __name__ == "__main__":
    main()
