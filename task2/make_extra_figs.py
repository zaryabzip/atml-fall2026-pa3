# extra task 2 figures. untrained vs trained accuracy and nll, and pair advantages with intervals.
# python task2/make_extra_figs.py --run artifacts/task2/main

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["correct", "complementary", "mismatched", "no_image"]
SUITES = ["vqa_test", "grounding_test"]


def load_results(path):
    return json.load(open(path))


def bar_with_ci(ax, labels, values, lows, highs, color, width, offset):
    x = np.arange(len(labels)) + offset
    err = [np.array(values) - np.array(lows), np.array(highs) - np.array(values)]
    ax.bar(x, values, width, color=color, yerr=err, capsize=3, error_kw={"lw": 0.8})


def fig_untrained_vs_trained(run, out):
    trained = load_results(run / "eval" / "results.json")
    untrained = load_results(run / "eval_untrained" / "results.json")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    width = 0.38
    for col, suite in enumerate(SUITES):
        labels = ["correct", "complementary", "mismatched", "no image"]
        for row, key in enumerate(["accuracy", "reference_nll"]):
            ax = axes[row, col]
            t = [trained[suite]["overall"][c][key] for c in CONDITIONS]
            u = [untrained[suite]["overall"][c][key] for c in CONDITIONS]
            tci = [trained[suite]["overall"][c][key + "_ci"] for c in CONDITIONS]
            uci = [untrained[suite]["overall"][c][key + "_ci"] for c in CONDITIONS]
            bar_with_ci(ax, labels, t, [c[0] for c in tci], [c[1] for c in tci], "#1f77b4", width, -width / 2)
            bar_with_ci(ax, labels, u, [c[0] for c in uci], [c[1] for c in uci], "#999999", width, width / 2)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=15, fontsize=9)
            ax.set_title(f"{suite}: {'accuracy' if key == 'accuracy' else 'reference NLL'}")
            ax.set_ylabel("accuracy" if key == "accuracy" else "reference-answer NLL")
            if key == "accuracy":
                ax.set_ylim(0, 1)
            if row == 0 and col == 0:
                ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
                                   plt.Rectangle((0, 0), 1, 1, color="#999999")],
                          labels=["trained (4500 updates)", "untrained (update 0)"], fontsize=8)
    fig.suptitle("task 2 test suites: untrained model vs trained model (95% intervals)")
    fig.tight_layout()
    fig.savefig(out / "fig_untrained_vs_trained.png", dpi=150)
    plt.close(fig)


def parse_ci(cell):
    m = re.match(r"\s*(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", cell)
    return tuple(float(x) for x in m.groups())


def fig_pair_advantage(run, out):
    rows = list(csv.DictReader(open(run / "report" / "table_advantage.csv")))
    labels = [f"{r['suite']}: {r['scope']}" for r in rows]
    acc = [parse_ci(r["accuracy_correct_minus_complementary"]) for r in rows]
    pair = [parse_ci(r["pair_accuracy_advantage"]) for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, vals, title in [(axes[0], acc, "accuracy: correct minus complementary"),
                            (axes[1], pair, "pair accuracy advantage")]:
        mid = [v[0] for v in vals]
        lo = [v[1] for v in vals]
        hi = [v[2] for v in vals]
        ax.errorbar(mid, y, xerr=[np.array(mid) - np.array(lo), np.array(hi) - np.array(mid)],
                    fmt="o", color="#1f77b4", capsize=3)
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("difference in proportion (0.2 = 20 points)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("does the correct image help? differences with 95% pair-bootstrap intervals")
    fig.tight_layout()
    fig.savefig(out / "fig_pair_advantage.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "report"
    out.mkdir(parents=True, exist_ok=True)
    fig_untrained_vs_trained(run, out)
    fig_pair_advantage(run, out)
    print("written:", sorted(p.name for p in out.glob("fig_*.png") if p.name in
                             ("fig_untrained_vs_trained.png", "fig_pair_advantage.png")))


if __name__ == "__main__":
    main()
