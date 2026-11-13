# writes the task 2 latex tables into report/tables/ from the saved results.
# rates are shown in percent here; the saved csv and json files keep fractions
# python task2/make_report_tables.py

import csv
import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from tabkit import ci, num, table  # noqa: E402

RES = R / "results" / "task2" / "main"
OUT = R / "report" / "tables"

SUITE = {"vqa_test": "VQAv2", "grounding_test": "Grounding"}
COND = {"correct": "correct", "complementary": "complementary", "mismatched": "mismatched", "no_image": "no image",
        "no image": "no image"}


def sci(x):
    mant, exp = f"{float(x):.0e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def read_csv(name):
    return list(csv.DictReader(open(RES / "report" / name)))


def parse(s):
    # "0.601 [0.531, 0.667]" to three floats
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]
    return v if len(v) == 3 else (v + [None, None])[:3]


def pct_ci(s):
    v, lo, hi = parse(s)
    return f"{num(100 * v)} {ci(100 * lo, 100 * hi)}"


def val_ci(s, fmt=".2f"):
    v, lo, hi = parse(s)
    return f"{num(v, fmt)} {ci(lo, hi, fmt)}"


cfg = json.load(open(RES / "config.json"))
stage_res = {r["stage"]: r for r in read_csv("table_resources_stage.csv")}
overview = read_csv("table_overview.csv")
untrained = json.load(open(RES / "eval_untrained" / "results.json"))

# 1. stages
rows = []
for s in cfg["stages"]:
    res = stage_res[s["name"]]
    mix = ", ".join(f"{k} {round(v * 100)}" for k, v in s["mix"].items())
    rows.append([s["name"], f"{s['updates']:,}", sci(s["conn_lr"]) if s["conn_lr"] else "--",
                 sci(s["lora_lr"]) if s["lora_lr"] else "--", mix,
                 f"{int(res['connector_params']):,}", f"{int(res['lora_params']):,}",
                 num(res["mean_seconds_per_update"], ".2f")])
OUT.joinpath("t2_stages.tex").write_text(table(
    "Training stages for Task 2. Mix is the percentage of each example type drawn in that stage. Seconds per update is the mean over the stage.",
    "tab:t2_stages",
    ["Stage", "Updates", r"Connector\\LR", r"LoRA\\LR", "Mix (%)", r"Connector\\params", r"LoRA\\params", r"s per\\update"],
    rows, r"lrrrp{3.4cm}rrr"))

# 2. resources. generation time per answer is a column of the overview table
tot = {r["item"]: r["value"] for r in read_csv("table_resources_total.csv")}
rows = [["Training", "wall time", f"{float(tot['train wall seconds']) / 3600:.1f} h ({float(tot['train wall seconds']):,.0f} s)"],
        ["Training", "peak MPS memory, sampled after each backward", f"{float(tot['peak mps memory GB (sampled after each backward)']):.2f} GB"],
        ["Feature cache", "images", f"{int(float(tot['feature cache images'])):,}"],
        ["Feature cache", "time", f"{float(tot['feature cache seconds']):,.1f} s"],
        ["Feature cache", "size", f"{float(tot['feature cache GB']):.2f} GB"]]
OUT.joinpath("t2_resources.tex").write_text(table(
    "Resources for the Task 2 main run and its feature cache.",
    "tab:t2_resources", ["Item", "Value"], rows, "lr", sections=True))

# 3. dev trajectory, with a stage column and rules at the stage boundaries
dev = read_csv("table_dev_trajectory.csv")
keys = ["caption_dev_loss", "grounding_nll_correct", "grounding_nll_complementary", "grounding_nll_no_image",
        "grounding_swap_penalty", "vqa_nll_correct", "vqa_nll_complementary", "vqa_nll_no_image", "vqa_swap_penalty"]
ends = {s["name"]: e for s, e in zip(cfg["stages"], [2250, 3750, 4500])}
starts = {"caption": 1, "instruct": 2251, "vqa": 3751}
stage_of = lambda u: "before training" if u == 0 else next(
    f"stage {i + 1}, {n} (updates {starts[n]:,} to {e:,})" for i, (n, e) in enumerate(ends.items()) if u <= e)
rows = [[stage_of(int(r["update"])), f"{int(r['update']):,}"] + [num(r[k], ".2f") for k in keys] for r in dev]
OUT.joinpath("t2_dev_trajectory.tex").write_text(table(
    "Development metrics during training. NLL is per answer. Swap penalty is partner NLL minus correct NLL, so positive values favor the correct image. Update 0 is before training.",
    "tab:t2_dev",
    ["Update", r"Caption\\dev loss", "Correct", "Compl.", "No image", "Swap", "Correct", "Compl.", "No image", "Swap"],
    rows, "r" * 10, sections=True, top=[("", 2), ("Grounding answer NLL", 4), ("VQAv2 answer NLL", 4)]))

# 4. overall test results
rows = [[f"{SUITE[r['suite']]}, {r['n_items']} questions", COND[r["condition"]], pct_ci(r["accuracy"]), pct_ci(r["pair_accuracy"]),
         val_ci(r["reference_nll"]), num(100 * float(r["identical_answer_rate"])), num(100 * float(r["empty_answer_rate"])),
         num(r["seconds_per_answer"], ".3f")] for r in overview]
OUT.joinpath("t2_overview.tex").write_text(table(
    "Test results for the trained model, by suite and image condition. Brackets are 95\\% bootstrap intervals. Pair accuracy marks a pair as correct only if both answers are correct.",
    "tab:t2_overview",
    ["Condition", r"Accuracy\\(\%)", r"Pair accuracy\\(\%)", r"Reference\\NLL", r"Identical\\answers (\%)",
     r"Empty\\answers (\%)", r"s per\\answer"],
    rows, "llllrrr", sections=True, size=r"\setlength{\tabcolsep}{4.5pt}"))

# 5. untrained versus trained
rows = []
trained = {(r["suite"], COND[r["condition"]]): r for r in overview}
for suite in ["vqa_test", "grounding_test"]:
    for cond_key in ["correct", "complementary", "mismatched", "no_image"]:
        u = untrained[suite]["overall"][cond_key]
        t = trained[(suite, COND[cond_key])]
        rows.append([SUITE[suite], COND[cond_key], f"{num(100 * u['accuracy'])} {ci(100 * u['accuracy_ci'][0], 100 * u['accuracy_ci'][1])}",
                     pct_ci(t["accuracy"]), f"{num(u['reference_nll'], '.2f')} {ci(*u['reference_nll_ci'], fmt='.2f')}",
                     val_ci(t["reference_nll"])])
OUT.joinpath("t2_untrained.tex").write_text(table(
    "Untrained model (update 0) against the trained model on the test suites. Brackets are 95\\% bootstrap intervals. The untrained model was rebuilt from the same seed and passed a dev check against the update-0 row of the training log.",
    "tab:t2_untrained",
    ["Condition", "Untrained", "Trained", "Untrained", "Trained"],
    rows, "lllll", sections=True, top=[("", 1), ("Accuracy (\\%)", 2), ("Reference NLL", 2)]))

# 6. correct minus complementary
adv = read_csv("table_advantage.csv")
rows = [[SUITE[r["suite"]], r["scope"], pct_ci(r["accuracy_correct_minus_complementary"]),
         pct_ci(r["pair_accuracy_advantage"]), val_ci(r["nll_complementary_minus_correct"])] for r in adv]
OUT.joinpath("t2_advantage.tex").write_text(table(
    "Effect of the correct image: correct minus complementary image, by suite and question type. Brackets are 95\\% pair-bootstrap intervals.",
    "tab:t2_advantage",
    ["Scope", r"Accuracy\\advantage (pp)", r"Pair accuracy\\advantage (pp)", r"NLL\\advantage"],
    rows, "lrrr", sections=True))

# 7. complementary behaviour
comp = read_csv("table_complementary.csv")
rows = [[SUITE[r["suite"]], r["scope"], num(100 * float(r["follows_partner_rate"])), num(100 * float(r["unsupported_rate"]))]
        for r in comp]
OUT.joinpath("t2_complementary.tex").write_text(table(
    "How often the model's answer follows the complementary image's reference answer, and how often it gives an answer the image does not support.",
    "tab:t2_complementary",
    ["Scope", r"Follows partner\\(\%)", r"Unsupported\\(\%)"],
    rows, "lrr", sections=True))

# 8. by question type
bytype = read_csv("table_by_type.csv")
rows = [[SUITE[r["suite"]], r["type"], COND[r["condition"]], r["n_items"], pct_ci(r["accuracy"]), pct_ci(r["pair_accuracy"]),
         val_ci(r["reference_nll"])] for r in bytype]
OUT.joinpath("t2_by_type.tex").write_text(table(
    "Test results by question type and image condition. Brackets are 95\\% bootstrap intervals.",
    "tab:t2_by_type",
    ["Type", "Condition", "Items", r"Accuracy\\(\%)", r"Pair accuracy\\(\%)", r"Reference\\NLL"],
    rows, "llrlll", group=1, sections=True))

print("wrote", sorted(p.name for p in OUT.glob("t2_*.tex")))
