# shared loading for task 3. supplied rollouts, splits, orders and per rollout scores.
# gold answers are loaded separately (load_gold) so selection code never sees them.

import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "artifacts" / "task3" / "data" / "ATML-PA3-Data"
MATHVISTA_META = ROOT / "artifacts" / "task3" / "mathvista" / "testmini.json"
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
SEED = 6304
N_ROLL = 32
N_ORDERS = 10
BUDGETS = [1, 2, 4, 8, 16, 32]
N_DEV = 200
N_LIVE = 50


def splits(question_ids):
    # sort lexicographically, permute with default_rng(6304), first 200 dev, rest eval
    ids = sorted(question_ids)
    perm = np.random.default_rng(SEED).permutation(len(ids))
    order = [ids[j] for j in perm]
    return {"permuted": order, "dev": order[:N_DEV], "eval": order[N_DEV:], "live_math": order[N_DEV:N_DEV + N_LIVE]}


def orders(question_ids, dataset_index):
    # ten seeded permutations of the 32 rollout ids per question, from a stream separate from the split
    rng = np.random.default_rng([SEED, dataset_index])
    return {q: np.stack([rng.permutation(N_ROLL) for _ in range(N_ORDERS)]) for q in sorted(question_ids)}


def _flat(table, col):
    arr = table.column(col).combine_chunks()
    return arr.values.to_numpy(zero_copy_only=False).astype(np.float64), arr.offsets.to_numpy()


def load(ds):
    # per rollout table indexed [question, rollout] plus flat token arrays with start/end per rollout
    folder = DATA / FOLDERS[ds]
    gen = pq.read_table(folder / "generations.parquet",
                        columns=["sample_id", "rollout_id", "extracted_answer", "n_generated_text_tokens",
                                 "hit_max_new_tokens"])
    prob = pq.read_table(folder / "probability_metrics.parquet", columns=["sample_id", "rollout_id", "logp", "kl_u_p"])
    sid = gen.column("sample_id").to_pylist()
    rid = gen.column("rollout_id").to_pylist()
    assert sid == prob.column("sample_id").to_pylist() and rid == prob.column("rollout_id").to_pylist()
    qids = sorted(set(sid))
    qpos = {q: i for i, q in enumerate(qids)}
    row = np.full((len(qids), N_ROLL), -1, dtype=np.int64)
    for r, (q, k) in enumerate(zip(sid, rid)):
        row[qpos[q], k] = r
    assert (row >= 0).all(), "missing rollouts"
    logp, off = _flat(prob, "logp")
    kl, off2 = _flat(prob, "kl_u_p")
    assert (off == off2).all()
    n_tok = np.array(gen.column("n_generated_text_tokens").to_pylist(), dtype=np.int64)
    assert (np.diff(off) == n_tok).all(), "token lists do not match lengths"
    out = {"qids": qids, "row": row, "start": off[:-1], "end": off[1:], "n_tok": n_tok,
           "capped": np.array(gen.column("hit_max_new_tokens").to_pylist(), dtype=bool),
           "extracted": gen.column("extracted_answer").to_pylist(), "logp": logp, "kl": kl}
    if ds == "mathvista":
        att = pq.read_table(folder / "attention_metrics.parquet",
                            columns=["sample_id", "rollout_id", "img_attn_total", "prompt_text_attn_total",
                                     "generated_text_attn_total"])
        assert sid == att.column("sample_id").to_pylist() and rid == att.column("rollout_id").to_pylist()
        img, o1 = _flat(att, "img_attn_total")
        prm, o2 = _flat(att, "prompt_text_attn_total")
        gtx, o3 = _flat(att, "generated_text_attn_total")
        assert (o1 == off).all() and (o2 == off).all() and (o3 == off).all()
        den = img + prm + gtx
        out["v"] = np.where(den > 0, img / np.where(den > 0, den, 1.0), 0.0)  # share of attention on the image
        out["v_zero_den"] = int((den <= 0).sum())
    return out


def rollout_means(flat, start, end):
    # mean of a flat token array per rollout. rollouts with no tokens get nan
    n = end - start
    sums = np.add.reduceat(flat, start[n > 0]) if (n > 0).any() else np.array([])
    out = np.full(len(start), np.nan)
    out[n > 0] = sums / n[n > 0]
    return out


def load_gold(ds):
    # gold answers by question id. kept apart from the selection inputs
    gen = pq.read_table(DATA / FOLDERS[ds] / "generations.parquet", columns=["sample_id", "gold_answer"])
    gold = {}
    for q, g in zip(gen.column("sample_id").to_pylist(), gen.column("gold_answer").to_pylist()):
        assert gold.setdefault(q, g) == g, f"gold differs across rollouts for {q}"
    return gold


def load_mathvista_meta():
    return json.load(open(MATHVISTA_META))
