# task 3 step 3 illustrative trajectories: token confidence c_t and visual engagement v_t over generation
# selection rule, fixed before plotting: the first held-out mathvista question (split order, order 0, n = 32) where
# self-consistency and visconf + rwc pick different answers. in each of the two winning answer groups, the rollouts
# with the highest and the lowest visconf score Z
# python task3/trajectories.py --results results/task3 --figures report/figures

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common
import offline

SMOOTH = 16


def smooth(x, k=SMOOTH):
    if len(x) < k:
        return np.asarray(x, float)
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    return (c[k:] - c[:-k]) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--figures", required=True)
    args = ap.parse_args()
    res, figs = Path(args.results), Path(args.figures)
    R = json.load(open(res / "offline" / "offline_results.json"))
    case = R["mathvista"]["cases"]["selection_disagreement"][0]
    D = offline.build("mathvista")
    orders = common.orders(D["qids"], 1)
    qi = D["qids"].index(case["question"])
    cand = orders[case["question"]][0][:32]
    C, Z, Zt = offline.visconf(cand, qi, D)
    sc, _ = offline.vote(cand, D["lab"][qi])
    rz, _ = offline.rwc(cand, D["lab"][qi], Z)
    name = lambda g: D["keys"][qi].get(int(g), "NO_ANSWER")
    assert name(sc) == case["self_consistency"] and name(rz) == case["rwc_Z"], "does not match the saved case"
    G = offline.grades("mathvista", D)  # gold for retrospective labels only

    # token level c_t with the same pooled 5th and 95th percentiles as visconf
    rows = D["row"][qi, cand]
    pool = np.concatenate([D["kl"][D["start"][r]:D["end"][r]] for r in rows])
    q05, q95 = np.quantile(pool, [0.05, 0.95])
    picks = []
    for role, g in [("self-consistency winner", sc), ("VisConf + RWC winner", rz)]:
        members = [r for r in cand if D["lab"][qi][r] == g]
        z = np.array([Z[r] for r in members])
        picks += [(role, "highest Z", members[int(np.nanargmax(z))]), (role, "lowest Z", members[int(np.nanargmin(z))])]
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import figstyle as fs
    fs.apply()
    out = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    shown = {"self-consistency winner": 0, "VisConf + RWC winner": 1}
    for role, which, r in picks:
        row = D["row"][qi, r]
        a, b = D["start"][row], D["end"][row]
        c = np.clip((D["kl"][a:b] - q05) / max(q95 - q05, 1e-8), 0, 1)
        v = D["v"][a:b]
        ok = offline.correct_of(qi, D["lab"][qi][r], G)
        rec = {"rollout": int(r), "role": role, "pick": which, "answer": name(D["lab"][qi][r]), "correct": bool(ok),
               "tokens": int(b - a), "C": float(C[r]), "V": float(D["V"][qi, r]), "Z": float(Z[r]), "Ztok": float(Zt[r]),
               "corr_c_v": float(np.corrcoef(c, v)[0, 1]) if len(c) > 2 and c.std() > 0 and v.std() > 0 else None}
        out.append(rec)
        if which != "highest Z":
            continue  # the figure shows the highest-Z rollout of each winning group
        ax = axes[shown[role]]
        x = np.arange(len(smooth(c))) + SMOOTH // 2
        ax.plot(x, smooth(c), color=fs.BLUE, label="confidence")
        ax.plot(x, smooth(v), color=fs.RED, label="image attention")
        ax.set_title(("Majority-vote answer" if shown[role] == 0 else "VisConf + RWC answer") + (" (correct)" if ok else " (wrong)"))
        ax.set_xlabel("generated token")
    axes[0].set_ylabel("value (smoothed)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(figs / "fig_t3_trajectories.png", dpi=150)
    plt.close(fig)
    (res / "offline" / "trajectories.json").write_text(json.dumps(
        {"question": case["question"], "rule": "first held-out question where self-consistency and VisConf + RWC differ "
                                               "(order 0, n = 32); highest and lowest Z rollout in each winning group",
         "group_sizes": case["group_sizes"], "calibration_q05_q95": [float(q05), float(q95)], "rollouts": out}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
