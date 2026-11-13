# task 3 rq2 checks on the saved rollouts (no model run): do wrong answers scatter, why does the vote miss a correct
# answer, and is the top-scoring rollout often wrong. held-out 800 questions, all 32 rollouts, order 0 breaks ties
# python task3/rq2_checks.py   writes results/task3/offline/rq2_checks.json
import json
from pathlib import Path

import numpy as np

import common
import offline

OUT = Path(__file__).resolve().parent.parent / "results" / "task3" / "offline" / "rq2_checks.json"
res = {}
for di, ds in enumerate(["math", "mathvista"]):
    D = offline.build(ds)
    G = offline.grades(ds, D)
    split = common.splits(D["qids"])
    held = set(split["heldout"]) if "heldout" in split else set(split["permuted"][200:])
    orders = common.orders(D["qids"], di)
    NA = offline.NA
    n_both = n_q = 0
    wrong_counts, wrong_groups, top_wrong_share, plural_correct, has_correct_q = [], [], [], 0, 0
    fail = []  # selection failures of the vote: (correct votes, winner votes, rank of correct group)
    top_ok = top_wrong_vote_right = top_wrong_vote_wrong = top_wrong_same = 0
    for qi, q in enumerate(D["qids"]):
        if q not in held:
            continue
        n_q += 1
        cand = orders[q][0][:32]
        lab = D["lab"][qi]
        corr = np.array([offline.correct_of(qi, lab[r], G) for r in cand])
        votes = {}
        for r in cand:
            votes[lab[r]] = votes.get(lab[r], 0) + 1
        # wrong rollouts: scatter over answer groups (no answer counts as one group of its own)
        wr = [lab[r] for r, c in zip(cand, corr) if not c]
        if corr.any() and wr:
            n_both += 1
            wrong_counts.append(len(wr))
            wrong_groups.append(len(set(wr)))
            top_wrong_share.append(max(wr.count(g) for g in set(wr)) / len(wr))
        sc, tot = offline.vote(cand, lab)
        if corr.any():
            has_correct_q += 1
            cg = [g for g in tot if offline.correct_of(qi, g, G)][0]
            ranked = sorted(tot, key=lambda g: -tot[g])
            if offline.correct_of(qi, sc, G):
                plural_correct += 1
            else:
                fail.append((tot[cg], tot[sc], ranked.index(cg) + 1))
        # top-scoring rollout: self-certainty on math, visconf on mathvista
        score = D["K"][qi] if ds == "math" else offline.visconf(cand, qi, D)[1]
        top = lab[cand[int(np.argmax(np.nan_to_num(score[cand], nan=-np.inf)))]]
        if offline.correct_of(qi, top, G):
            top_ok += 1
        else:
            if offline.correct_of(qi, sc, G):
                top_wrong_vote_right += 1
            else:
                top_wrong_vote_wrong += 1
                top_wrong_same += int(top == sc)
    f = np.array(fail)
    res[ds] = {
        "questions": n_q, "questions_with_correct_and_wrong": n_both,
        "mean_wrong_rollouts": float(np.mean(wrong_counts)), "mean_distinct_wrong_answers": float(np.mean(wrong_groups)),
        "mean_share_of_wrong_in_largest_wrong_group": float(np.mean(top_wrong_share)),
        "questions_with_a_correct_rollout": has_correct_q, "vote_correct_given_correct_exists": plural_correct / has_correct_q,
        "vote_misses_correct": len(fail),
        "miss_mean_correct_votes": float(f[:, 0].mean()), "miss_mean_winner_votes": float(f[:, 1].mean()),
        "miss_correct_rank_share": {str(k): float((f[:, 2] == k).mean()) for k in (2, 3)},
        "miss_correct_votes_at_most_4": float((f[:, 0] <= 4).mean()),
        "top_score_correct": top_ok / n_q, "top_wrong_vote_right": top_wrong_vote_right / n_q,
        "top_wrong_vote_wrong": top_wrong_vote_wrong / n_q,
        "top_wrong_vote_wrong_same_answer_share": top_wrong_same / max(top_wrong_vote_wrong, 1),
    }
    print(ds, json.dumps(res[ds], indent=1), flush=True)
OUT.write_text(json.dumps(res, indent=1))
