# task 3 figures and latex tables from the saved offline (and, if present, live) results.
# python task3/report_task3.py --results results/task3 --tables report/tables --figures report/figures

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabkit import TODO_CAPTION, num, table  # noqa: E402

NAMES = {"first": "FirstAns", "sc": "Self-consistency", "best_L": "Highest log-prob", "best_K": "Highest self-certainty",
         "rwc_K": "Self-certainty + RWC", "shortest": "Shortest rollout", "best_V": "Vision only, highest",
         "rwc_V": "Vision only + RWC", "best_Z": "VisConf, highest", "rwc_Z": "VisConf + RWC",
         "best_Ztok": "Tokenwise fusion, highest", "rwc_Ztok": "Tokenwise fusion + RWC"}
DS = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
BUDGETS = ["1", "2", "4", "8", "16", "32"]


def pct(x):
    return f"{100 * x:.1f}"


def methods_of(res):
    return [m for m in NAMES if m in res["eval"]["by_budget"]["1"]]


def fig_curves(R, out):
    # plain version: coverage, first answer, majority vote, the best single score, and the score-weighted vote
    import figstyle as fs
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, ds in zip(axes, ["math", "mathvista"]):
        bb = R[ds]["eval"]["by_budget"]
        x = [int(n) for n in BUDGETS]
        best, rwc = ("best_K", "rwc_K") if ds == "math" else ("best_Z", "rwc_Z")
        lines = [("Oracle coverage", [100 * bb[n]["oracle"] for n in BUDGETS], fs.GRAY, "--"),
                 ("Score-weighted vote", [100 * bb[n][rwc]["accuracy"] for n in BUDGETS], fs.GREEN, "-"),
                 ("Majority vote", [100 * bb[n]["sc"]["accuracy"] for n in BUDGETS], fs.BLUE, "-"),
                 ("Highest score", [100 * bb[n][best]["accuracy"] for n in BUDGETS], fs.ORANGE, "-"),
                 ("First answer", [100 * bb[n]["first"]["accuracy"] for n in BUDGETS], fs.LIGHT, "-")]
        for name, y, color, ls in lines:
            ax.plot(x, y, color=color, ls=ls, marker="o", ms=3)
        lo, hi = min(min(l[1]) for l in lines), max(max(l[1]) for l in lines)
        fs.end_labels(ax, [(l[1][-1], l[0], l[2] if l[2] != fs.LIGHT else fs.GRAY) for l in lines], 36, (hi - lo) * 0.075)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.set_xlabel("candidates N")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(DS[ds])
        ax.set_xlim(0.9, 32)
    fig.subplots_adjust(left=0.07, right=0.86, wspace=0.62, bottom=0.17, top=0.9)
    fig.savefig(out / "fig_t3_accuracy_vs_n.png", dpi=150)
    plt.close(fig)


def fig_cost(R, out):
    # plain version: the two selection rules that matter and the two adaptive policies
    import figstyle as fs
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, ds in zip(axes, ["math", "mathvista"]):
        bb = R[ds]["eval"]["by_budget"]
        rwc = "rwc_K" if ds == "math" else "rwc_Z"
        for m, name, color in [("sc", "Majority vote", fs.BLUE), (rwc, "Score-weighted vote", fs.GREEN)]:
            cost = [bb[n]["cost_all_n"] for n in BUDGETS]
            ax.plot(cost, [100 * bb[n][m]["accuracy"] for n in BUDGETS], color=color, marker="o", ms=3, label=name)
        ad = R[ds]["adaptive"]
        for fam, mk, label in [("agreement", "s", "Adaptive, agreement"), ("quality", "^", "Adaptive, quality")]:
            h = ad["families"][fam]["held_out"]
            ax.plot([h["mean_tokens"]], [100 * h["accuracy"]], mk, color=fs.ORANGE, ms=8, label=label, ls="none")
        ax.legend(fontsize=8.5, loc="lower right")
        ax.set_xscale("log")
        ax.set_xlabel("generated tokens per question")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(DS[ds])
    fig.subplots_adjust(left=0.07, right=0.98, wspace=0.28, bottom=0.17, top=0.9)
    fig.savefig(out / "fig_t3_accuracy_vs_cost.png", dpi=150)
    plt.close(fig)


def plain(s):
    # text with every latex special character escaped
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "^": r"\^{}",
           "_": r"\_", "%": r"\%", "~": r"\~{}"}
    return "".join(rep.get(c, c) for c in str(s))


def balanced(s):
    depth = 0
    for c in s:
        depth += {"{": 1, "}": -1}.get(c, 0)
        if depth < 0:
            return False
    return depth == 0


def math_answer(s):
    # a math answer typeset as math, split at top-level commas so long lists can wrap
    parts, depth, cur = [], 0, ""
    for c in s:
        depth += {"{": 1, "}": -1}.get(c, 0)
        if c == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += c
    parts.append(cur.strip())
    if not all(p and balanced(p) for p in parts):
        return plain(s)
    return ", ".join(f"${p}$" for p in parts)


def answer(ds, a):
    if a in (None, "NO_ANSWER"):
        return "no answer"
    if ds == "math":
        return math_answer(a)
    kind, _, val = a.partition(":")
    if kind == "CHOICE":
        return f"({val})"
    if kind == "UNPARSED":
        return plain(val) + " (unparsed)"
    return plain(val) if kind in ("NUM", "LIST") else plain(a)


def mark(ok):
    return "right" if ok else "wrong"


def tables_offline(R, out):
    for ds in ["math", "mathvista"]:
        bb = R[ds]["eval"]["by_budget"]
        rows = [["Oracle coverage"] + [pct(bb[n]["oracle"]) for n in BUDGETS],
                ["Distinct answer groups"] + [f"{bb[n]['distinct_groups']:.2f}" for n in BUDGETS]]
        rows += [[NAMES[m]] + [pct(bb[n][m]["accuracy"]) for n in BUDGETS] for m in methods_of(R[ds])]
        out.joinpath(f"t3_{ds}_methods.tex").write_text(table(
            f"{DS[ds]}: accuracy (\\%) by method and candidate budget N on the held-out 800 questions, averaged over ten orders.",
            f"tab:t3_{ds}_methods", ["Method"] + [f"$N={n}$" for n in BUDGETS], rows, "l" + "r" * 6, rules_after=[1]))
        # one row per method, both budgets side by side
        boot = R[ds]["eval"]["bootstrap"]
        rows = []
        for m in methods_of(R[ds]):
            if m == "sc":
                continue
            row = [NAMES[m]]
            for n in ["8", "32"]:
                d = boot[f"{m}_minus_sc@{n}"]
                row += [num(100 * d["mean"], sign=True), f"[{num(100 * d['ci95'][0], sign=True)}, {num(100 * d['ci95'][1], sign=True)}]"]
            rows.append(row)
        out.joinpath(f"t3_{ds}_bootstrap.tex").write_text(table(
            f"{DS[ds]}: accuracy difference from self-consistency in percentage points, with 95\\% intervals from 1,000 paired question-level bootstrap resamples.",
            f"tab:t3_{ds}_bootstrap", ["Method", "Difference", "95\\% interval", "Difference", "95\\% interval"], rows,
            "lrrrr", top=[("", 1), ("$N=8$", 2), ("$N=32$", 2)]))
        # at N=32 each method's questions split into correct, a correct candidate missed, and no correct candidate
        rows = [[NAMES[m], pct(bb["32"][m]["accuracy"]), pct(bb["32"][m]["selection_failure"]), pct(bb["32"]["sc"]["sampling_failure"])]
                for m in methods_of(R[ds])]
        out.joinpath(f"t3_{ds}_failures.tex").write_text(table(
            TODO_CAPTION, f"tab:t3_{ds}_failures",
            ["Method", "Correct", r"Selection\\failure", r"Sampling\\failure"],
            rows, "lrrr", top=[("", 1), (r"\% of question-order pairs at $N=32$", 3)]))


SCORES = [("length", "Rollout length"), ("L", "Mean log-probability"), ("K", "Self-certainty"),
          ("V", "Visual engagement"), ("Z", "VisConf"), ("Ztok", "Tokenwise fusion")]


def tables_signals(R, out):
    key = lambda ds, k: R[ds]["signals"].get(f"top_minus_bottom_quartile_{k}")
    rows = [[name] + [num(100 * key(ds, k), sign=True) if key(ds, k) is not None else "--" for ds in ["math", "mathvista"]]
            for k, name in SCORES]
    out.joinpath("t3_signals.tex").write_text(table(
        "Within-question association with correctness at N=32: correctness in the top score quartile minus the bottom quartile, in percentage points, averaged over held-out questions.",
        "tab:t3_signals", ["Score", DS["math"], DS["mathvista"]], rows, "lrr"))


def tables_adaptive(R, out):
    rows = []
    for ds in ["math", "mathvista"]:
        ad = R[ds]["adaptive"]
        for n in ["4", "8", "16", "32"]:
            f = ad["fixed_sc"]["eval"][n]
            rows.append([DS[ds], f"Fixed, $N={n}$", "--", pct(f["accuracy"]), f"{f['mean_tokens']:,.0f}", f"{int(n):.1f}"]
                        + ["100" if m == n else "--" for m in ["4", "8", "16", "32"]])
        for fam in ["agreement", "quality"]:
            F = ad["families"][fam]
            h, sel = F["held_out"], F["selected"]
            setting = "always 32" if sel["tau"] is None else (f"$\\tau={sel['tau']}$" + (f", $\\gamma={sel['gamma']:g}$" if sel["gamma"] is not None else ""))
            rows.append([DS[ds], f"Adaptive, {fam}", setting, pct(h["accuracy"]), f"{h['mean_tokens']:,.0f}",
                         f"{h['mean_rollouts']:.1f}"] + [f"{100 * h['stop_distribution'].get(m, 0):.0f}" for m in ["4", "8", "16", "32"]])
    out.joinpath("t3_adaptive.tex").write_text(table(
        "Adaptive stopping on the held-out 800 questions. Settings were chosen on the 200 development questions: lowest token cost within one point of fixed-32 self-consistency. Stop distribution is the share of decisions stopping at 4, 8, 16 or 32 candidates.",
        "tab:t3_adaptive", ["Policy", "Setting", r"Accuracy\\(\%)", r"Mean\\tokens", r"Mean\\rollouts",
                            "4", "8", "16", "32"],
        rows, "llrrrrrrr", sections=True, top=[("", 5), (r"Stops at $n$ (\% of decisions)", 4)]))


def tables_signals_extra(R, out):
    rows = []
    for ds in ["math", "mathvista"]:
        c, w, n = R[ds]["signals"]["mean_length_correct_vs_wrong"]
        rows += [[DS[ds], "Mean tokens, correct rollouts", f"{c:.0f}"], [DS[ds], "Mean tokens, wrong rollouts", f"{w:.0f}"],
                 [DS[ds], "Questions with both correct and wrong rollouts", f"{n}"]]
    cv = R["mathvista"]["signals"]["confidence_vision_corr"]
    gap = R["mathvista"]["signals"]["tokenwise_minus_trajectory_equals_cov_max_err"]
    mant, exp = f"{gap:.1e}".split("e")
    rows += [[DS["mathvista"], "Within-rollout corr$(c_t, v_t)$, mean over questions", num(cv["mean"], ".3f")],
             [DS["mathvista"], "Within-rollout corr$(c_t, v_t)$, median over questions", num(cv["median"], ".3f")],
             [DS["mathvista"], f"Questions with positive corr$(c_t, v_t)$ (\\% of {cv['questions']})", num(100 * cv["share_positive"])],
             [DS["mathvista"], r"$\max_i|Z^{tok}_i-Z_i-\mathrm{Cov}_t(c,v)|$", f"${mant}\\times10^{{{int(exp)}}}$"]]
    bb = R["mathvista"]["eval"]["by_budget"]["32"]
    rows += [[DS["mathvista"], f"Accuracy at $N=32$ (\\%), {NAMES[m]}", pct(bb[m]["accuracy"])]
             for m in ["best_Z", "best_Ztok", "rwc_Z", "rwc_Ztok"]]
    out.joinpath("t3_signals_extra.tex").write_text(table(TODO_CAPTION, "tab:t3_signals_extra",
                                                          ["Measure", "Value"], rows, "lr", sections=True))


def tables_early_stops(R, out):
    rows = []
    for ds in ["math", "mathvista"]:
        bb = R[ds]["eval"]["by_budget"]
        for fam in ["agreement", "quality"]:
            F = R[ds]["adaptive"]["families"][fam]
            e, h = F["early_stop_analysis"], F["held_out"]
            if not e:
                continue
            # fixed-n self-consistency closest in mean token cost
            n = min(BUDGETS, key=lambda b: abs(bb[b]["cost_all_n"] - h["mean_tokens"]))
            rows.append([DS[ds], fam, f"{e['early_stops']:,}", f"{e['wrong_early_stops']:,}",
                         f"{e['wrong_early_where_fixed32_correct']:,}", pct(h["accuracy"]), f"{h['mean_tokens']:,.0f}",
                         n, pct(bb[n]["sc"]["accuracy"]), f"{bb[n]['cost_all_n']:,.0f}"])
    out.joinpath("t3_early_stops.tex").write_text(table(
        TODO_CAPTION, "tab:t3_early_stops",
        ["Policy", "All", "Wrong", r"Wrong, fixed-32\\SC right", r"Accuracy\\(\%)", "Tokens", "$N$",
         r"Accuracy\\(\%)", "Tokens"], rows, "lrrrrrrrr", sections=True,
        top=[("", 1), ("Early stops (of 8,000)", 3), ("Adaptive", 2), ("Closest-cost fixed SC", 3)]))


def tables_step6(R, out):
    # one table per dataset, ten rows each
    for ds in ["math", "mathvista"]:
        rows = []
        for fam in ["agreement", "quality"]:
            s = R[ds]["step6"].get(fam)
            if not s:
                continue
            for b, v in s.items():
                if v["decisions"]:
                    rows.append([fam, b if b == "0" else "$" + b.replace(",", ", ") + "$", f"{v['decisions']:,}", f"{v['mean_rollouts']:.1f}", pct(v["early_stop_share"]),
                                 pct(v["adaptive_accuracy"]), pct(v["fixed32_sc_accuracy"])])
        out.joinpath(f"t3_step6_{ds}.tex").write_text(table(
            TODO_CAPTION, f"tab:t3_step6_{ds}",
            ["Policy", r"Correct share\\of the 32", "Decisions", r"Mean\\rollouts", r"Early stop\\(\%)",
             r"Adaptive\\accuracy (\%)", r"Fixed-32 SC\\accuracy (\%)"], rows, "llrrrrr", group=1))
    rows = []
    for ds in ["math", "mathvista"]:
        for fam in ["agreement", "quality"]:
            s = R[ds]["step6"].get(fam + "_by_agreement32")
            if s:
                rows.append([DS[ds], fam] + [f"{s[k]['mean_rollouts']:.1f} ({s[k]['decisions']:,})" for k in ["<=0.5", "(0.5,0.75]", ">0.75"]])
    out.joinpath("t3_step6_agreement.tex").write_text(table(
        TODO_CAPTION, "tab:t3_step6_agreement",
        ["Policy", "$\\le 0.5$", "$(0.5, 0.75]$", "$> 0.75$"], rows, "lrrr", sections=True,
        top=[("", 1), (r"\begin{tabular}[b]{@{}c@{}}Mean rollouts (decisions)\\by top-answer share of the 32\end{tabular}", 3)]))


def tables_cases(R, out):
    sel, stop = [], []
    for ds in ["math", "mathvista"]:
        cs = R[ds]["cases"]
        rq = "rwc_Z" if ds == "mathvista" else "rwc_K"
        for c in cs["selection_disagreement"]:
            top = sorted(c["group_sizes"].items(), key=lambda kv: -kv[1])[:3]
            sel.append([DS[ds], plain(c["question"]), f"{answer(ds, c['self_consistency'])} ({mark(c['sc_correct'])})",
                        f"{answer(ds, c[rq])} ({mark(c[rq + '_correct'])})",
                        "; ".join(f"{answer(ds, k)}: {v}" for k, v in top)])
        for c in cs["premature_stops"]:
            stop.append([DS[ds], plain(c["question"]), str(c["stopped_at"]), answer(ds, c["stopped_answer"]),
                         answer(ds, c["fixed32_answer"])])
    out.joinpath("t3_cases_selection.tex").write_text(table(
        TODO_CAPTION, "tab:t3_cases_selection",
        ["Question", r"Self-consistency\\answer", r"Score + RWC\\answer", "Largest answer groups (votes of 32)"],
        sel, r"lp{2.9cm}p{3.6cm}p{5.0cm}", sections=True, size=r"\small"))
    out.joinpath("t3_cases_stops.tex").write_text(table(
        TODO_CAPTION, "tab:t3_cases_stops",
        ["Question", r"Stopped\\at", r"Answer at stop\\(wrong)", r"Fixed-32 SC answer\\(right)"],
        stop, "lrll", sections=True))


def tables_live_cases(P, out):
    # stated rule: in live question order, the first wrong to correct and the first correct to wrong revision,
    # and the first two questions whose initial response has no boxed answer
    rows = []
    for kind, test, k in [("wrong to correct", lambda p: not p["initial_correct"] and p["refinement_correct"], 1),
                          ("correct to wrong", lambda p: p["initial_correct"] and not p["refinement_correct"], 1),
                          ("initial without box", lambda p: p["parallel_answers"][0] is None, 2)]:
        for p in [p for p in P if test(p)][:k]:
            rows.append([kind, plain(p["question"]),
                         "none (cap)" if p["parallel_answers"][0] is None else answer("math", p["parallel_answers"][0]),
                         f"{answer('math', p['revised_answer'])} ({mark(p['refinement_correct'])})",
                         f"{answer('math', p['single_answer'])} ({mark(p['single_correct'])})",
                         f"{answer('math', p['parallel_selected'])} ({mark(p['parallel_correct'])})"])
    out.joinpath("t3_live_cases.tex").write_text(table(
        TODO_CAPTION, "tab:t3_live_cases", ["Case", "Question", "Initial", "Revised", "Single", "Parallel SC"],
        rows, "llllll", group=1))
    # paired comparison on the same 50 questions
    rows = []
    for a, b in [("single", "parallel"), ("single", "refinement"), ("parallel", "refinement")]:
        xa, xb = [p[f"{a}_correct"] for p in P], [p[f"{b}_correct"] for p in P]
        d = [int(y) - int(x) for x, y in zip(xa, xb)]
        m = sum(d) / len(d)
        se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
        rows.append([f"{a} to {b}", sum(x and y for x, y in zip(xa, xb)), sum(x and not y for x, y in zip(xa, xb)),
                     sum(y and not x for x, y in zip(xa, xb)), f"{num(100 * m, sign=True)} $\\pm$ {num(100 * se)}"])
    out.joinpath("t3_live_paired.tex").write_text(table(
        TODO_CAPTION, "tab:t3_live_paired", ["Comparison", r"Both\\right", r"Only first\\right", r"Only second\\right",
                                             "Difference (pp)"], rows, "lrrrr"))


def tables_live(L, out):
    rows = []
    for s, name in [("single", "Single longer response"), ("parallel", "Parallel sampling (3 x 1,024)"),
                    ("refinement", "Sequential refinement")]:
        v = L["strategies"][s]
        rows.append([name, pct(v["accuracy"]), f"{v['mean_input_tokens']:,.0f}", f"{v['mean_output_tokens']:,.0f}",
                     f"{v['mean_seconds']:.1f}", pct(v["truncation_rate_per_call"]), pct(v["no_boxed_rate_per_call"])])
    t = L["revision_transitions"]
    out.joinpath("t3_live.tex").write_text(table(
        f"Live comparison on the 50 fixed MATH questions with the same 3,072-token output allowance. Refinement transitions relative to the shared initial answer: wrong to correct {t['wrong_to_correct']}, correct to wrong {t['correct_to_wrong']}, stayed correct {t['same_correct']}, stayed wrong {t['same_wrong']}.",
        "tab:t3_live", ["Strategy", r"Accuracy\\(\%)", r"Input\\tokens", r"Output\\tokens", "Seconds",
                        r"Truncated\\(\% of calls)", r"No box\\(\% of calls)"],
        rows, "lrrrrrr", top=[("", 1), ("Mean per question", 4), ("", 2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--figures", required=True)
    args = ap.parse_args()
    res, tab, figs = Path(args.results), Path(args.tables), Path(args.figures)
    tab.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    R = json.load(open(res / "offline" / "offline_results.json"))
    fig_curves(R, figs)
    fig_cost(R, figs)
    tables_offline(R, tab)
    tables_signals(R, tab)
    tables_adaptive(R, tab)
    tables_signals_extra(R, tab)
    tables_early_stops(R, tab)
    tables_step6(R, tab)
    tables_cases(R, tab)
    live = res / "live" / "live_summary.json"
    if live.exists():
        tables_live(json.load(open(live)), tab)
        tables_live_cases(json.load(open(res / "live" / "live_per_question.json")), tab)
    print("task 3 tables and figures written")


if __name__ == "__main__":
    main()
