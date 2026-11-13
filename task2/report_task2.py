# task 2 report. builds figures tables and qualitative examples from one run dir.
# run with python report_task2.py --run <run dir>

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# paths and settings

ROOT = Path(__file__).resolve().parent.parent / "data" / "task2"
SEED = 6304
CONDS = ["correct", "complementary", "mismatched", "no_image"]
COND_LABEL = {"correct": "correct", "complementary": "complementary", "mismatched": "mismatched",
              "no_image": "no image"}
SUITES = ["vqa_test", "grounding_test"]
SUITE_TYPES = {"vqa_test": ["yes/no", "number", "other"], "grounding_test": ["presence", "counting"]}
QUAL_PER_TYPE = 2


# io helpers

def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def last_by_update(rows):
    # resumed runs can log an update twice so keep the last copy
    d = {}
    for r in rows:
        d[r["update"]] = r
    return [d[k] for k in sorted(d)]


def write_table(name, header, rows, out):
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n")


def fmt(v, ci=None, digits=3):
    s = f"{v:.{digits}f}"
    if ci is not None:
        s += f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"
    return s


def stage_ends(config):
    ends, total = [], 0
    for s in config["stages"]:
        total += s["updates"]
        ends.append((s["name"], total))
    return ends


# figures

def plot_learning(train, dev, ends, out):
    # plain version: training loss, caption dev loss, answer NLL (correct image and no image) and the image-swap penalty,
    # stage boundaries in light grey
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import figstyle as fs
    fs.apply()
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
    ax = ax.ravel()
    u = np.array([r["update"] for r in train])
    loss = np.array([r["loss"] for r in train])
    k = min(25, len(loss))
    ax[0].plot(u[k - 1:], np.convolve(loss, np.ones(k) / k, mode="valid"), color=fs.BLUE)
    ax[0].set_title("Training loss")
    ax[0].set_ylabel("loss per target token")
    du = np.array([r["update"] for r in dev])
    ax[1].plot(du, [r["caption_dev_loss"] for r in dev], "o-", color=fs.BLUE, ms=4)
    ax[1].set_title("Caption loss on development set")
    ax[1].set_ylabel("loss per target token")
    for suite, colour, name in [("grounding_dev", fs.BLUE, "Grounding"), ("vqa_dev", fs.ORANGE, "VQAv2")]:
        ax[2].plot(du, [r[f"{suite}_nll_correct"] for r in dev], "o-", color=colour, ms=4, label=f"{name}, correct image")
        ax[2].plot(du, [r[f"{suite}_nll_no_image"] for r in dev], "o:", color=colour, ms=4, label=f"{name}, no image")
        ax[3].plot(du, [r[f"{suite}_swap_penalty"] for r in dev], "o-", color=colour, ms=4, label=name)
    ax[2].set_title("Answer NLL on development set")
    ax[2].set_ylabel("NLL per answer")
    ax[2].set_ylim(0, 18)
    ax[2].legend(fontsize=8, loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.8)
    ax[3].axhline(0, color="#888888", lw=0.8)
    ax[3].set_title("Image-swap penalty")
    ax[3].set_ylabel("NLL partner minus NLL correct")
    ax[3].legend(fontsize=9, loc="upper left")
    for a_ in ax:
        a_.set_xlabel("update")
        for _, end in ends[:-1]:
            a_.axvline(end, color="#c4c4c4", ls="--", lw=0.8)
    prev = 0
    for name, end in ends:
        ax[0].text((prev + end) / 2, ax[0].get_ylim()[1], name, fontsize=8.5, ha="center", va="top", color="#555555")
        prev = end
    fig.tight_layout()
    fig.savefig(out / "fig_learning_curves.png", dpi=150)
    plt.close(fig)


def plot_accuracy(results, out):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for col, suite in enumerate(SUITES):
        ov = results[suite]["overall"]
        xs = np.arange(len(CONDS))
        for row, (key, ci_key, title) in enumerate([("accuracy", "accuracy_ci", "accuracy"),
                                                    ("pair_accuracy", "pair_accuracy_ci", "pair accuracy")]):
            vals = np.array([ov[c][key] for c in CONDS])
            cis = np.array([ov[c][ci_key] for c in CONDS])
            err = np.vstack([vals - cis[:, 0], cis[:, 1] - vals])
            a = ax[row, col]
            a.bar(xs, vals, yerr=err, capsize=4, color=["tab:green", "tab:orange", "tab:gray", "tab:purple"])
            a.set_xticks(xs)
            a.set_xticklabels([COND_LABEL[c] for c in CONDS], rotation=15)
            a.set_title(f"{suite} {title}")
            a.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "fig_accuracy_by_condition.png", dpi=150)
    plt.close(fig)


def plot_by_type(results, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for col, suite in enumerate(SUITES):
        types = SUITE_TYPES[suite]
        xs = np.arange(len(types))
        width = 0.2
        for i, c in enumerate(CONDS):
            vals = [results[suite]["by_type"][t][c]["accuracy"] for t in types]
            cis = np.array([results[suite]["by_type"][t][c]["accuracy_ci"] for t in types])
            vals = np.array(vals)
            err = np.vstack([vals - cis[:, 0], cis[:, 1] - vals])
            ax[col].bar(xs + (i - 1.5) * width, vals, width, yerr=err, capsize=3, label=COND_LABEL[c])
        ax[col].set_xticks(xs)
        ax[col].set_xticklabels(types)
        ax[col].set_title(f"{suite} accuracy by type")
        ax[col].set_ylim(0, 1)
        ax[col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_accuracy_by_type.png", dpi=150)
    plt.close(fig)


# tables

def table_overview(results, out):
    header = ["suite", "condition", "n_items", "accuracy", "pair_accuracy", "reference_nll",
              "identical_answer_rate", "empty_answer_rate", "seconds_per_answer"]
    rows = []
    for suite in SUITES:
        ov = results[suite]["overall"]
        for c in CONDS:
            d = ov[c]
            rows.append([suite, COND_LABEL[c], d["n_items"], fmt(d["accuracy"], d["accuracy_ci"]),
                         fmt(d["pair_accuracy"], d["pair_accuracy_ci"]),
                         fmt(d["reference_nll"], d["reference_nll_ci"]),
                         fmt(d["identical_answer_rate"]), fmt(d["empty_answer_rate"]),
                         fmt(d["seconds_per_answer"], digits=3)])
    write_table("table_overview", header, rows, out)


def table_by_type(results, out):
    header = ["suite", "type", "condition", "n_items", "accuracy", "pair_accuracy", "reference_nll"]
    rows = []
    for suite in SUITES:
        for t in SUITE_TYPES[suite]:
            for c in CONDS:
                d = results[suite]["by_type"][t][c]
                rows.append([suite, t, COND_LABEL[c], d["n_items"], fmt(d["accuracy"], d["accuracy_ci"]),
                             fmt(d["pair_accuracy"], d["pair_accuracy_ci"]),
                             fmt(d["reference_nll"], d["reference_nll_ci"])])
    write_table("table_by_type", header, rows, out)


def table_advantage(results, out):
    header = ["suite", "scope", "accuracy_correct_minus_complementary", "pair_accuracy_advantage",
              "nll_complementary_minus_correct"]
    rows = []
    for suite in SUITES:
        scopes = [("overall", results[suite]["overall"])]
        scopes += [(t, results[suite]["by_type"][t]) for t in SUITE_TYPES[suite]]
        for name, blk in scopes:
            a = blk["advantage"]
            rows.append([suite, name, fmt(a["accuracy_correct_minus_complementary"], a["accuracy_advantage_ci"]),
                         fmt(a["pair_accuracy_advantage"], a["pair_accuracy_advantage_ci"]),
                         fmt(a["nll_complementary_minus_correct"], a["nll_advantage_ci"])])
    write_table("table_advantage", header, rows, out)


def table_complementary(results, out):
    header = ["suite", "scope", "follows_partner_rate", "unsupported_rate"]
    rows = []
    for suite in SUITES:
        scopes = [("overall", results[suite]["overall"])]
        scopes += [(t, results[suite]["by_type"][t]) for t in SUITE_TYPES[suite]]
        for name, blk in scopes:
            rows.append([suite, name, fmt(blk["complementary_follows_partner_rate"]),
                         fmt(blk["complementary_unsupported_rate"])])
    write_table("table_complementary", header, rows, out)


def table_dev(dev, out):
    header = ["update", "caption_dev_loss", "grounding_nll_correct", "grounding_nll_complementary",
              "grounding_nll_no_image", "grounding_swap_penalty", "vqa_nll_correct",
              "vqa_nll_complementary", "vqa_nll_no_image", "vqa_swap_penalty"]
    rows = [[r["update"], fmt(r["caption_dev_loss"]), fmt(r["grounding_dev_nll_correct"]),
             fmt(r["grounding_dev_nll_complementary"]), fmt(r["grounding_dev_nll_no_image"]),
             fmt(r["grounding_dev_swap_penalty"]), fmt(r["vqa_dev_nll_correct"]),
             fmt(r["vqa_dev_nll_complementary"]), fmt(r["vqa_dev_nll_no_image"]),
             fmt(r["vqa_dev_swap_penalty"])] for r in dev]
    write_table("table_dev_trajectory", header, rows, out)


def table_resources(run, train, summary, config, results, out):
    # per stage cost from the training log
    header = ["stage", "updates", "connector_params", "lora_params", "mean_seconds_per_update"]
    rows = []
    for name, end in stage_ends(config):
        info = next(s for s in summary["stages"] if s["stage"] == name) if summary.get("stages") else None
        seconds = [r["seconds"] for r in train if r["stage"] == name]
        mean_s = round(float(np.mean(seconds)), 2) if seconds else ""
        rows.append([name, sum(1 for r in train if r["stage"] == name),
                     info["connector_params"] if info else "",
                     info["lora_params"] if info else "", mean_s])
    write_table("table_resources_stage", header, rows, out)

    # whole run and feature cache costs
    cache_path = ROOT / "features" / "cache_meta.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else None
    header = ["item", "value"]
    rows = [["train wall seconds", summary.get("wall_seconds", "")],
            ["peak mps memory GB (sampled after each backward)", summary.get("peak_mem_GB_sampled", "")],
            ["feature cache images", cache["images"] if cache else "not recorded"],
            ["feature cache seconds", cache["seconds"] if cache else "not recorded"],
            ["feature cache GB", round(cache["bytes"] / 1e9, 2) if cache else "not recorded"]]
    for suite in SUITES:
        for c in CONDS:
            rows.append([f"{suite} {COND_LABEL[c]} seconds per answer",
                         round(results[suite]["overall"][c]["seconds_per_answer"], 3)])
    write_table("table_resources_total", header, rows, out)


# qualitative examples

def select_examples(suite, pairs, rng):
    chosen = []
    for t in SUITE_TYPES[suite]:
        pool = [p for p in pairs if p["type"] == t]
        idx = rng.choice(len(pool), size=min(QUAL_PER_TYPE, len(pool)), replace=False)
        chosen += [pool[int(i)] for i in idx]
    return chosen


def qualitative(run, suites, preds, out):
    rng = np.random.default_rng(SEED)
    qdir = out / "qualitative"
    qdir.mkdir(exist_ok=True)
    md = ["# qualitative examples", "",
          "selection rule: for each suite and answer type, draw two pairs uniformly at random "
          f"with seed {SEED} from the test pairs of that type. no filtering on correctness.", ""]
    for suite in SUITES:
        for p in select_examples(suite, suites[suite], rng):
            pid = p["pair_id"]
            a, b = p["items"]
            md += [f"## {suite} {pid} ({p['type']})", "", f"question: {p['question']}", ""]
            md += ["| item | reference | " + " | ".join(COND_LABEL[c] for c in CONDS) + " |",
                   "|---|---|" + "---|" * len(CONDS)]
            for k, it in enumerate(p["items"]):
                preds_row = [preds[(pid, k, c)]["pred"] for c in CONDS]
                md.append(f"| {'AB'[k]} | {it['answer']} | " + " | ".join(preds_row) + " |")
            md.append("")

            fig, ax = plt.subplots(1, 2, figsize=(10, 6))
            for k, it in enumerate(p["items"]):
                img = Image.open(ROOT / "coco" / it["file"]).convert("RGB")
                ax[k].imshow(img)
                ax[k].axis("off")
                lines = [f"{'AB'[k]} reference: {it['answer']}"]
                lines += [f"{COND_LABEL[c]}: {preds[(pid, k, c)]['pred']}" for c in CONDS]
                ax[k].set_title("\n".join(lines), fontsize=8)
            fig.suptitle(p["question"], fontsize=10)
            fig.tight_layout()
            fig.savefig(qdir / f"{suite}_{pid}.png", dpi=120)
            plt.close(fig)
    (out / "qualitative.md").write_text("\n".join(md) + "\n")


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "report"
    out.mkdir(parents=True, exist_ok=True)

    config = json.load(open(run / "config.json"))
    summary = json.load(open(run / "summary.json"))
    train = last_by_update(read_jsonl(run / "train_log.jsonl"))
    dev = last_by_update(read_jsonl(run / "dev_log.jsonl"))
    results = json.load(open(run / "eval" / "results.json"))
    preds_rows = read_jsonl(run / "eval" / "predictions.jsonl")
    preds = {(r["pair_id"], r["k"], r["cond"]): r for r in preds_rows}
    suites = {"vqa_test": read_jsonl(ROOT / "manifests" / "vqa_test.jsonl"),
              "grounding_test": read_jsonl(ROOT / "manifests" / "grounding_test.jsonl")}

    ends = stage_ends(config)
    plot_learning(train, dev, ends, out)
    plot_accuracy(results, out)
    plot_by_type(results, out)
    table_overview(results, out)
    table_by_type(results, out)
    table_advantage(results, out)
    table_complementary(results, out)
    table_dev(dev, out)
    table_resources(run, train, summary, config, results, out)
    qualitative(run, suites, preds, out)
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
