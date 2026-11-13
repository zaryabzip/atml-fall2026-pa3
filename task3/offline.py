# task 3 offline experiments on the supplied rollouts: steps 1, 2, 3 and 5.
# phase 1 builds answer groups and scores without gold answers. phase 2 runs every selection and stopping rule
# on the saved orders. phase 3 loads gold answers, grades, and summarizes.
# python task3/offline.py --out results/task3/offline

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import answers
import common

TAU = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMA = [0.0, 0.1, 0.25, 0.5]
CHECKS = [4, 8, 16]
NA = -1  # code of the NO_ANSWER group


def avg_rank_desc(x):
    # rank 1 is the largest score. tied scores share their average rank
    order = np.argsort(-x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def vote(cand, lab, weights=None):
    # summed votes per group. ties go to the group whose first supporting rollout is earliest in the order
    tot, first = {}, {}
    for pos, r in enumerate(cand):
        g = lab[r]
        tot[g] = tot.get(g, 0.0) + (1.0 if weights is None else weights[pos])
        first.setdefault(g, pos)
    best = max(tot, key=lambda g: (tot[g], -first[g]))
    return best, tot


def best_score(cand, lab, score):
    s = np.nan_to_num(score[cand], nan=-np.inf)
    return lab[cand[int(np.argmax(s))]]  # argmax keeps the earliest on ties


def rwc(cand, lab, score):
    n = len(cand)
    if n == 1:
        return lab[cand[0]], {lab[cand[0]]: 1.0}
    r = avg_rank_desc(np.nan_to_num(score[cand], nan=-np.inf))
    # w = (n - r) / (n - 1). the common factor 2 / (n - 1) is dropped so the weights are exact integers
    return vote(cand, lab, np.rint(2 * (n - r)))


def visconf(cand, q, D):
    # calibrate token kl on the pooled tokens of the current candidates only, then C_i, Z_i and tokenwise Z
    rows = D["row"][q, cand]
    segs = [(D["start"][r], D["end"][r]) for r in rows]
    pool = np.concatenate([D["kl"][a:b] for a, b in segs])
    q05, q95 = np.quantile(pool, [0.05, 0.95]) if len(pool) else (0.0, 0.0)
    den = max(q95 - q05, 1e-8)
    C, Zt = np.full(len(cand), np.nan), np.full(len(cand), np.nan)
    for i, (a, b) in enumerate(segs):
        if b > a:
            c = np.clip((D["kl"][a:b] - q05) / den, 0.0, 1.0)
            C[i] = c.mean()
            Zt[i] = (c * D["v"][a:b]).mean()
    V = D["V"][q, cand]
    full = lambda x: _scatter(x, cand)
    return full(C), full(V * C), full(Zt)


def _scatter(vals, cand):
    out = np.full(common.N_ROLL, np.nan)
    out[cand] = vals
    return out


# phase 1: groups and scores, no gold

def build(ds):
    D = common.load(ds)
    nq = len(D["qids"])
    meta = common.load_mathvista_meta() if ds == "mathvista" else None
    if meta is not None:
        meta = {k: {kk: vv for kk, vv in v.items() if kk != "answer"} for k, v in meta.items()}  # keep gold out
    lab = np.full((nq, common.N_ROLL), NA, dtype=np.int64)
    keys = []
    # answer groups are cached by a hash of the scorer code. they never use gold answers
    tag = hashlib.sha1(Path(answers.__file__).read_bytes()).hexdigest()[:10]
    cache = common.ROOT / "artifacts" / "task3" / "cache" / f"{ds}_groups_{tag}.json"
    if cache.exists():
        c = json.load(open(cache))
        D["lab"], D["keys"] = np.array(c["lab"], dtype=np.int64), [{int(k): v for k, v in d.items()} for d in c["keys"]]
        answers.SCORER_PATHS.update(c.get("paths", {}))
        return finish_build(ds, D)
    for qi, q in enumerate(D["qids"]):
        ext = [D["extracted"][D["row"][qi, r]] for r in range(common.N_ROLL)]
        if ds == "math":
            k = answers.group_math(ext)
        else:
            k = [answers.normalize_mathvista(e, meta[q]) for e in ext]
        codes = {}
        for r, key in enumerate(k):
            if key != answers.NO_ANSWER:
                lab[qi, r] = codes.setdefault(key, len(codes))
        keys.append({v: kk for kk, v in codes.items()})
    D["lab"], D["keys"] = lab, keys
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"lab": lab.tolist(), "keys": [{str(k): v for k, v in d.items()} for d in keys],
                                 "paths": dict(answers.SCORER_PATHS) if ds == "mathvista" else {}}))
    return finish_build(ds, D)


def finish_build(ds, D):
    D["T"] = D["n_tok"][D["row"]]
    D["L"] = common.rollout_means(D["logp"], D["start"], D["end"])[D["row"]]
    D["K"] = common.rollout_means(D["kl"], D["start"], D["end"])[D["row"]]
    if ds == "mathvista":
        D["V"] = common.rollout_means(D["v"], D["start"], D["end"])[D["row"]]
    D["zero_token_rollouts"] = int((D["T"] == 0).sum())
    return D


# phase 2: selections and stopping, still no gold

def methods_for(ds):
    m = ["first", "sc", "best_L", "best_K", "rwc_K", "shortest"]
    if ds == "mathvista":
        m += ["best_V", "rwc_V", "best_Z", "rwc_Z", "best_Ztok", "rwc_Ztok"]
    return m


def select_all(ds, D, orders):
    nq = len(D["qids"])
    meth = methods_for(ds)
    sel = {m: np.zeros((nq, common.N_ORDERS, len(common.BUDGETS)), dtype=np.int64) for m in meth}
    cost = np.zeros((nq, common.N_ORDERS, len(common.BUDGETS)))
    groups = np.zeros_like(cost)
    stop = {}  # (q, o, n) -> sc winner, unique lead count, rwc winner, rwc margin
    for qi, q in enumerate(D["qids"]):
        lab, T = D["lab"][qi], D["T"][qi]
        for o in range(common.N_ORDERS):
            perm = orders[q][o]
            for bi, n in enumerate(common.BUDGETS):
                cand = perm[:n]
                cost[qi, o, bi] = T[cand].sum()
                groups[qi, o, bi] = len(set(lab[cand]))
                sc, votes = vote(cand, lab)
                pick = {"first": lab[cand[0]], "sc": sc,
                        "best_L": best_score(cand, lab, D["L"][qi]),
                        "best_K": best_score(cand, lab, D["K"][qi]),
                        "rwc_K": rwc(cand, lab, D["K"][qi])[0],
                        "shortest": best_score(cand, lab, -T.astype(float))}
                if ds == "mathvista":
                    _, Z, Zt = visconf(cand, qi, D)
                    pick.update({"best_V": best_score(cand, lab, D["V"][qi]), "rwc_V": rwc(cand, lab, D["V"][qi])[0],
                                 "best_Z": best_score(cand, lab, Z), "rwc_Z": rwc(cand, lab, Z)[0],
                                 "best_Ztok": best_score(cand, lab, Zt), "rwc_Ztok": rwc(cand, lab, Zt)[0]})
                    qscore = Z
                else:
                    qscore = D["K"][qi]
                for m in meth:
                    sel[m][qi, o, bi] = pick[m]
                if n in CHECKS + [32]:
                    counts = sorted(votes.values(), reverse=True)
                    unique = len(counts) == 1 or counts[0] > counts[1]
                    rw, w = rwc(cand, lab, qscore)
                    ws = sorted(w.values(), reverse=True)
                    margin = (ws[0] - (ws[1] if len(ws) > 1 else 0.0)) / sum(ws) if sum(ws) > 0 else 0.0
                    stop[(qi, o, n)] = (sc, votes[sc] if unique else -1, rw, margin)
    return sel, cost, groups, stop


def adaptive(stop, D, qidx, tau, gamma=None):
    # returns chosen group, rollouts used and token cost per (question, order)
    out_g = np.zeros((len(qidx), common.N_ORDERS), dtype=np.int64)
    out_n = np.zeros_like(out_g)
    out_c = np.zeros((len(qidx), common.N_ORDERS))
    for i, qi in enumerate(qidx):
        for o in range(common.N_ORDERS):
            used = 32
            for n in CHECKS:
                sc, lead, rw, margin = stop[(qi, o, n)]
                ok = lead > 0 and sc != NA and lead / n >= tau
                if ok and gamma is not None:
                    ok = rw == sc and margin >= gamma
                if ok:
                    used = n
                    break
            out_g[i, o] = stop[(qi, o, used)][0]
            out_n[i, o] = used
    return out_g, out_n


# phase 3: grading and summaries

def grades(ds, D):
    gold = common.load_gold(ds)
    out = []
    if ds == "math":
        for qi, q in enumerate(D["qids"]):
            out.append({c: answers.grade_math(k, gold[q]) for c, k in D["keys"][qi].items()})
    else:
        meta = common.load_mathvista_meta()
        for qi, q in enumerate(D["qids"]):
            m = meta[q]
            if m["question_type"] == "multi_choice":
                # the parquet stores the option letter, testmini the option text. both must point at the same option
                letter = chr(ord("A") + m["choices"].index(m["answer"]))
                assert gold[q] in (letter, m["answer"]), f"gold option differs from testmini for {q}"
            else:
                assert gold[q] == m["answer"], f"parquet gold differs from testmini for {q}"
            gk = answers.gold_mathvista(m["answer"], m)
            out.append({c: answers.grade_mathvista(k, gk) for c, k in D["keys"][qi].items()})
    return out


def correct_of(qi, g, G):
    return g != NA and G[qi].get(int(g), False)


def bootstrap(diff, rng, n=1000):
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(ds, D, orders, sel, cost, groups, stop, G, split, out):
    qpos = {q: i for i, q in enumerate(D["qids"])}
    res = {}
    for part in ["dev", "eval"]:
        qidx = np.array([qpos[q] for q in split[part]])
        corr = {m: np.array([[[correct_of(qi, sel[m][qi, o, b], G) for b in range(len(common.BUDGETS))]
                              for o in range(common.N_ORDERS)] for qi in qidx]) for m in sel}
        orc = np.array([[[any(correct_of(qi, D["lab"][qi][r], G) for r in orders[D["qids"][qi]][o][:n])
                          for n in common.BUDGETS] for o in range(common.N_ORDERS)] for qi in qidx])
        for m in sel:  # checks: accuracy never above coverage, every rule agrees at n = 1
            assert not (corr[m] & ~orc).any(), f"{m} correct without a correct candidate"
            assert (sel[m][qidx, :, 0] == sel["first"][qidx, :, 0]).all(), f"{m} differs at n = 1"
        assert (np.diff(orc.mean(axis=1), axis=1) >= -1e-12).all(), "coverage decreased with n"
        per_q = {m: corr[m].mean(axis=1) for m in sel}  # average over orders within a question
        table = {}
        for bi, n in enumerate(common.BUDGETS):
            row = {"oracle": float(orc[:, :, bi].mean()), "distinct_groups": float(groups[qidx, :, bi].mean()),
                   "cost_all_n": float(cost[qidx, :, bi].mean()), "cost_first": float(cost[qidx, :, 0].mean())}
            for m in sel:
                wrong = ~corr[m][:, :, bi]
                row[m] = {"accuracy": float(per_q[m][:, bi].mean()),
                          "sampling_failure": float((wrong & ~orc[:, :, bi]).mean()),
                          "selection_failure": float((wrong & orc[:, :, bi]).mean())}
            table[n] = row
        rng = np.random.default_rng(common.SEED)
        diffs = {}
        for bi, n in enumerate(common.BUDGETS):
            for m in sel:
                if m != "sc":
                    d = per_q[m][:, bi] - per_q["sc"][:, bi]
                    diffs[f"{m}_minus_sc@{n}"] = {"mean": float(d.mean()), "ci95": bootstrap(d, rng)}
            for s in ["K", "V", "Z", "Ztok"]:
                if f"rwc_{s}" in sel:
                    d = per_q[f"rwc_{s}"][:, bi] - per_q[f"best_{s}"][:, bi]
                    diffs[f"rwc_{s}_minus_best_{s}@{n}"] = {"mean": float(d.mean()), "ci95": bootstrap(d, rng)}
        res[part] = {"by_budget": table, "bootstrap": diffs}
        save_per_question(out / f"{ds}_{part}_per_question.csv", D, qidx, per_q, orc, cost)
    res["adaptive"] = run_adaptive(ds, D, stop, G, split, qpos, cost)
    res["signals"] = signals(ds, D, G, [qpos[q] for q in split["eval"]])
    res["step6"] = step6(ds, D, stop, G, split, qpos, res["adaptive"])
    res["cases"] = cases(ds, D, orders, sel, stop, G, split, qpos, res["adaptive"])
    meth = list(sel)
    np.savez(out / f"{ds}_selections.npz", qids=np.array(D["qids"]), methods=np.array(meth),
             budgets=np.array(common.BUDGETS), selected_group=np.stack([sel[m] for m in meth]))
    (out / f"{ds}_answer_groups.json").write_text(json.dumps(
        [{str(c): k for c, k in d.items()} for d in D["keys"]]))
    return res


def save_per_question(path, D, qidx, per_q, orc, cost):
    with open(path, "w") as f:
        f.write("question,budget,method,accuracy,oracle,cost\n")
        for i, qi in enumerate(qidx):
            for bi, n in enumerate(common.BUDGETS):
                for m, a in per_q.items():
                    c = cost[qi, :, 0].mean() if m == "first" else cost[qi, :, bi].mean()  # firstans uses one candidate
                    f.write(f"{D['qids'][qi]},{n},{m},{a[i, bi]:.4f},{orc[i, :, bi].mean():.4f},{c:.1f}\n")


def run_adaptive(ds, D, stop, G, split, qpos, cost):
    T = D["T"]
    def evaluate(part, tau, gamma):
        qidx = [qpos[q] for q in split[part]]
        g, n_used = adaptive(stop, D, qidx, tau, gamma)
        acc = np.array([[correct_of(qi, g[i, o], G) for o in range(common.N_ORDERS)] for i, qi in enumerate(qidx)])
        tok = np.array([[T[qi][common_orders[D["qids"][qi]][o][:n_used[i, o]]].sum() for o in range(common.N_ORDERS)]
                        for i, qi in enumerate(qidx)])
        return {"accuracy": float(acc.mean(axis=1).mean()), "mean_tokens": float(tok.mean()),
                "mean_rollouts": float(n_used.mean()),
                "stop_distribution": {str(n): float((n_used == n).mean()) for n in CHECKS + [32]},
                "_acc": acc, "_n": n_used, "_g": g}

    def fixed_sc(part, n):
        qidx = [qpos[q] for q in split[part]]
        bi = common.BUDGETS.index(n)
        acc = np.array([[correct_of(qi, stop[(qi, o, n)][0], G) for o in range(common.N_ORDERS)] for qi in qidx])
        return float(acc.mean(axis=1).mean()), float(cost[qidx, :, bi].mean())

    sc32_dev, _ = fixed_sc("dev", 32)
    out = {"fixed_sc": {part: {str(n): dict(zip(["accuracy", "mean_tokens"], fixed_sc(part, n))) for n in CHECKS + [32]}
                        for part in ["dev", "eval"]}, "families": {}}
    for family in ["agreement", "quality"]:
        grid = []
        for tau in TAU:
            for gamma in (GAMMA if family == "quality" else [None]):
                r = evaluate("dev", tau, gamma)
                grid.append({"tau": tau, "gamma": gamma, **{k: v for k, v in r.items() if not k.startswith("_")}})
        grid.append({"tau": None, "gamma": None, "accuracy": sc32_dev,
                     "mean_tokens": out["fixed_sc"]["dev"]["32"]["mean_tokens"], "mean_rollouts": 32.0,
                     "stop_distribution": {"32": 1.0}, "fallback_always_32": True})
        ok = [g for g in grid if g["accuracy"] >= sc32_dev - 0.01 - 1e-12]
        pick = min(ok, key=lambda g: (g["mean_tokens"], -g["accuracy"],
                                      g["tau"] if g["tau"] is not None else 9, g["gamma"] if g["gamma"] is not None else 9))
        if pick.get("fallback_always_32"):
            held = {"accuracy": out["fixed_sc"]["eval"]["32"]["accuracy"],
                    "mean_tokens": out["fixed_sc"]["eval"]["32"]["mean_tokens"], "mean_rollouts": 32.0,
                    "stop_distribution": {"32": 1.0}}
            early = {}
        else:
            r = evaluate("eval", pick["tau"], pick["gamma"])
            held = {k: v for k, v in r.items() if not k.startswith("_")}
            qidx = [qpos[q] for q in split["eval"]]
            sc32 = np.array([[correct_of(qi, stop[(qi, o, 32)][0], G) for o in range(common.N_ORDERS)] for qi in qidx])
            wrong_early = (~r["_acc"]) & (r["_n"] < 32)
            early = {"wrong_early_stops": int(wrong_early.sum()),
                     "wrong_early_where_fixed32_correct": int((wrong_early & sc32).sum()),
                     "early_stops": int((r["_n"] < 32).sum()), "decisions": int(r["_n"].size)}
        out["families"][family] = {"dev_grid": grid, "selected": {"tau": pick["tau"], "gamma": pick["gamma"]},
                                   "held_out": held, "early_stop_analysis": early}
    return out


def signals(ds, D, G, qidx):
    # n = 32, all rollouts. top minus bottom quartile correctness per score within a question, averaged over questions
    res, vis_corr, fusion_gap = {}, [], 0.0
    scores = {"length": lambda qi: D["T"][qi].astype(float), "L": lambda qi: D["L"][qi], "K": lambda qi: D["K"][qi]}
    full = np.arange(common.N_ROLL)
    cache = {}
    if ds == "mathvista":
        for qi in qidx:
            C, Z, Zt = visconf(full, qi, D)
            cache[qi] = (Z, Zt)
        scores.update({"V": lambda qi: D["V"][qi], "Z": lambda qi: cache[qi][0], "Ztok": lambda qi: cache[qi][1]})
    for name, f in scores.items():
        diffs = []
        for qi in qidx:
            corr = np.array([correct_of(qi, D["lab"][qi][r], G) for r in full], dtype=float)
            s = np.nan_to_num(f(qi), nan=-np.inf)
            order = np.argsort(-s, kind="mergesort")
            diffs.append(corr[order[:8]].mean() - corr[order[-8:]].mean())
        res[f"top_minus_bottom_quartile_{name}"] = float(np.mean(diffs))
    lens_c, lens_w = [], []
    for qi in qidx:
        corr = np.array([correct_of(qi, D["lab"][qi][r], G) for r in full])
        if corr.any() and (~corr).any():
            lens_c.append(D["T"][qi][corr].mean())
            lens_w.append(D["T"][qi][~corr].mean())
    res["mean_length_correct_vs_wrong"] = [float(np.mean(lens_c)), float(np.mean(lens_w)), len(lens_c)]
    if ds == "mathvista":
        per_q = []
        for qi in qidx:
            rows = D["row"][qi]
            pool = np.concatenate([D["kl"][D["start"][r]:D["end"][r]] for r in rows])
            q05, q95 = np.quantile(pool, [0.05, 0.95])
            den = max(q95 - q05, 1e-8)
            rs = []
            for r in rows:
                a, b = D["start"][r], D["end"][r]
                c = np.clip((D["kl"][a:b] - q05) / den, 0, 1)
                v = D["v"][a:b]
                if b - a >= 3 and c.std() > 0 and v.std() > 0:
                    rs.append(float(np.corrcoef(c, v)[0, 1]))
                if b > a:
                    gap = (c * v).mean() - c.mean() * v.mean() - np.cov(c, v, bias=True)[0, 1] if b - a > 1 else 0.0
                    fusion_gap = max(fusion_gap, abs(float(gap)))
            if rs:
                per_q.append(np.mean(rs))
        res["confidence_vision_corr"] = {"mean": float(np.mean(per_q)), "median": float(np.median(per_q)),
                                         "share_positive": float(np.mean(np.array(per_q) > 0)), "questions": len(per_q)}
        res["tokenwise_minus_trajectory_equals_cov_max_err"] = fusion_gap
    return res


def step6(ds, D, stop, G, split, qpos, adaptive_res):
    # retrospective only: how stopping relates to agreement and to the share of correct rollouts in the full pool
    qidx = [qpos[q] for q in split["eval"]]
    out = {}
    for family in ["agreement", "quality"]:
        sel = adaptive_res["families"][family]["selected"]
        if sel["tau"] is None:
            continue
        g, n_used = adaptive(stop, D, qidx, sel["tau"], sel["gamma"])
        bins = {"0": [], "(0,0.25]": [], "(0.25,0.5]": [], "(0.5,0.75]": [], "(0.75,1]": []}
        for i, qi in enumerate(qidx):
            frac = np.mean([correct_of(qi, D["lab"][qi][r], G) for r in range(common.N_ROLL)])
            b = "0" if frac == 0 else "(0,0.25]" if frac <= 0.25 else "(0.25,0.5]" if frac <= 0.5 else "(0.5,0.75]" if frac <= 0.75 else "(0.75,1]"
            for o in range(common.N_ORDERS):
                bins[b].append((n_used[i, o], correct_of(qi, g[i, o], G), correct_of(qi, stop[(qi, o, 32)][0], G)))
        out[family] = {b: {"decisions": len(v), "mean_rollouts": float(np.mean([x[0] for x in v])) if v else None,
                           "early_stop_share": float(np.mean([x[0] < 32 for x in v])) if v else None,
                           "adaptive_accuracy": float(np.mean([x[1] for x in v])) if v else None,
                           "fixed32_sc_accuracy": float(np.mean([x[2] for x in v])) if v else None} for b, v in bins.items()}
        # stopping against agreement among all 32 candidates
        agree = {"<=0.5": [], "(0.5,0.75]": [], ">0.75": []}
        for i, qi in enumerate(qidx):
            counts = np.bincount(D["lab"][qi] - D["lab"][qi].min())
            share = counts.max() / common.N_ROLL
            k = "<=0.5" if share <= 0.5 else "(0.5,0.75]" if share <= 0.75 else ">0.75"
            agree[k] += list(n_used[i])
        out[family + "_by_agreement32"] = {k: {"decisions": len(v), "mean_rollouts": float(np.mean(v)) if v else None}
                                           for k, v in agree.items()}
    return out


def cases(ds, D, orders, sel, stop, G, split, qpos, adaptive_res):
    # stated rule: the first eval questions in split order (order 0, n = 32) where self-consistency and the
    # rwc rule with the dataset's quality score pick different answers, and the first early wrong stops of the
    # selected agreement policy where fixed-32 self-consistency is correct
    rq = "rwc_Z" if ds == "mathvista" else "rwc_K"
    bi = common.BUDGETS.index(32)
    keys = D["keys"]
    name = lambda qi, g: keys[qi].get(int(g), answers.NO_ANSWER) if g != NA else answers.NO_ANSWER
    disagree, early = [], []
    for q in split["eval"]:
        qi = qpos[q]
        a, b = sel["sc"][qi, 0, bi], sel[rq][qi, 0, bi]
        if a != b and len(disagree) < 3:
            disagree.append({"question": q, "self_consistency": name(qi, a), "sc_correct": correct_of(qi, a, G),
                             rq: name(qi, b), f"{rq}_correct": correct_of(qi, b, G),
                             "group_sizes": {name(qi, g): int((D["lab"][qi] == g).sum()) for g in set(D["lab"][qi])}})
    sel_a = adaptive_res["families"]["agreement"]["selected"]
    if sel_a["tau"] is not None:
        for q in split["eval"]:
            qi = qpos[q]
            g, n_used = adaptive(stop, D, [qi], sel_a["tau"], None)
            if n_used[0, 0] < 32 and not correct_of(qi, g[0, 0], G) and correct_of(qi, stop[(qi, 0, 32)][0], G):
                early.append({"question": q, "stopped_at": int(n_used[0, 0]), "stopped_answer": name(qi, g[0, 0]),
                              "fixed32_answer": name(qi, stop[(qi, 0, 32)][0])})
            if len(early) == 3:
                break
    return {"selection_disagreement": disagree, "premature_stops": early, "rule": "first eval questions in split order, order 0"}


def no_peek_check(D, orders):
    # changing an unrevealed rollout's token kl must not change visconf on the revealed prefix
    qi, q = 0, D["qids"][0]
    cand = orders[q][0][:4]
    before = visconf(cand, qi, D)[1][cand].copy()
    hidden = orders[q][0][5]
    r = D["row"][qi, hidden]
    saved = D["kl"][D["start"][r]:D["end"][r]].copy()
    D["kl"][D["start"][r]:D["end"][r]] += 100.0
    after = visconf(cand, qi, D)[1][cand]
    D["kl"][D["start"][r]:D["end"][r]] = saved
    return bool(np.allclose(before, after, equal_nan=True))


common_orders = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"scorer": {"math_verify": __import__("importlib.metadata").metadata.version("math-verify"),
                         "mathvista_rules": "official normalize_extracted_answer, plus formatting-only cleanup when float() fails"}}
    for di, ds in enumerate(["math", "mathvista"]):
        t0 = time.time()
        D = build(ds)
        split = common.splits(D["qids"])
        assert split["live_math"] == split["permuted"][200:250]
        orders = common.orders(D["qids"], di)
        global common_orders
        common_orders = orders
        (out / f"{ds}_splits.json").write_text(json.dumps({k: v for k, v in split.items() if k != "permuted"}, indent=1))
        np.savez(out / f"{ds}_orders.npz", qids=np.array(D["qids"]), orders=np.stack([orders[q] for q in D["qids"]]))
        sel, cost, groups, stop = select_all(ds, D, orders)
        # cost at every budget equals the token sum of the first n candidates. firstans uses only the first
        q0 = D["qids"][0]
        assert cost[0, 0, 3] == D["T"][0][orders[q0][0][:8]].sum() and cost[0, 0, 0] == D["T"][0][orders[q0][0][0]]
        t_select = time.time() - t0
        G = grades(ds, D)  # gold enters only here
        report[ds] = summarize(ds, D, orders, sel, cost, groups, stop, G, split, out)
        report[ds]["checks"] = {"zero_token_rollouts": D["zero_token_rollouts"], "seconds_to_select": round(t_select, 1),
                                "visconf_no_peek": no_peek_check(D, orders) if ds == "mathvista" else None,
                                "attention_zero_denominators": D.get("v_zero_den")}
        print(ds, "done in", round(time.time() - t0, 1), "s", flush=True)
    report["scorer"]["mathvista_paths"] = dict(answers.SCORER_PATHS)
    (out / "offline_results.json").write_text(json.dumps(report, indent=1, default=str))
    print("written", out / "offline_results.json")


if __name__ == "__main__":
    main()
