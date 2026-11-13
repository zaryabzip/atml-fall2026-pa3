# task 2 eval on held out test suites
# greedy decoding 16 tokens batch 8

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from prep_data import normalize_answer
from train_vlm import Features, read_jsonl
from vlm import ANSWER_CAP, VLM

# spec settings

SEED = 6304
ROOT = Path(__file__).resolve().parent.parent / "data" / "task2"
CONDITIONS = ["correct", "complementary", "mismatched", "no_image"]
BATCH = 8
MAX_NEW = 16
N_BOOT = 4000
VQA_TYPES = ["yes/no", "number", "other"]
GROUND_TYPES = ["presence", "counting"]


# items

def make_items(suite, pairs, mismatch):
    # one record per item per condition
    items = []
    for p in pairs:
        for k, it in enumerate(p["items"]):
            partner = p["items"][1 - k]
            image_for = {"correct": it["image_id"], "complementary": partner["image_id"],
                         "mismatched": mismatch[f"{p['pair_id']}|{k}"], "no_image": None}
            for cond in CONDITIONS:
                items.append({"suite": suite, "pair_id": p["pair_id"], "k": k, "type": p["type"],
                              "question": p["question"], "cond": cond, "image_id": image_for[cond],
                              "answer": it["answer"], "answer_norm": it["answer_norm"],
                              "answers_all": it["answers_all"],
                              "partner_answer_norm": partner["answer_norm"]})
    return items


# scoring

def soft_accuracy(pred_norm, answers_all):
    # vqa soft accuracy over 10 annotators
    norms = [normalize_answer(a) for a in answers_all]
    scores = []
    for i in range(len(norms)):
        others = sum(1 for j, a in enumerate(norms) if j != i and a == pred_norm)
        scores.append(min(1.0, others / 3.0))
    return float(np.mean(scores))


# generation and nll

@torch.no_grad()
def run_batch(vlm, feats, recs, with_image):
    feat = feats.get([r["image_id"] for r in recs]) if with_image else None
    gen_parts, nll_parts = [], []
    for r in recs:
        prefix, suffix, _ = vlm.build_example(None, r["question"], "", ANSWER_CAP)
        gen_parts.append((prefix, suffix, []))
        _, _, a_ids = vlm.build_example(None, r["question"], r["answer"], ANSWER_CAP)
        nll_parts.append((prefix, suffix, a_ids[:-1]))

    # left pad for batched generation
    t0 = time.time()
    with torch.autocast(vlm.device.type, dtype=torch.float16):
        embeds, mask, _ = vlm.assemble(feat, gen_parts, with_image=with_image, left_pad=True, targets=False)
        out = vlm.core.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=MAX_NEW,
                                do_sample=False, use_cache=True, eos_token_id=vlm.im_end,
                                pad_token_id=vlm.tok.pad_token_id)
    # split wall time across batch
    gen_seconds = (time.time() - t0) / len(recs)
    texts = vlm.tok.batch_decode(out, skip_special_tokens=True)

    # teacher forced nll
    with torch.autocast(vlm.device.type, dtype=torch.float16):
        e2, m2, l2 = vlm.assemble(feat, nll_parts, with_image=with_image, left_pad=False, targets=True)
        nll = vlm.example_nll(e2, m2, l2).tolist()
    return texts, nll, gen_seconds


def score_items(suite, items):
    for r in items:
        r["pred"] = r["pred_text"].strip()
        r["pred_norm"] = normalize_answer(r["pred"])
        r["canonical_match"] = r["pred_norm"] == r["answer_norm"]
        # grounding exact match vqa soft accuracy
        if suite == "grounding_test":
            r["metric"] = float(r["canonical_match"])
        else:
            r["metric"] = soft_accuracy(r["pred_norm"], r["answers_all"])
        # empty outputs stay in denominators
        r["empty"] = r["pred_norm"] == ""


# bootstrap

def boot_idx(n, rng, subset=None):
    # resample pairs with replacement
    if subset is None:
        return rng.integers(0, n, size=(N_BOOT, n))
    return subset[rng.integers(0, len(subset), size=(N_BOOT, len(subset)))]


def ci(values, idx):
    # 95 percent interval
    return np.percentile(values[idx].mean(axis=1), [2.5, 97.5]).tolist()


# summaries

def summarize(suite, pairs, items, rng):
    types = GROUND_TYPES if suite == "grounding_test" else VQA_TYPES
    by_pair = {}
    for r in items:
        by_pair.setdefault(r["pair_id"], {}).setdefault(r["cond"], []).append(r)

    pair_ids = [p["pair_id"] for p in pairs]
    pair_type = np.array([p["type"] for p in pairs])
    # same pair order in every condition
    per = {c: {"metric": [], "pair_ok": [], "nll": [], "identical": []} for c in CONDITIONS}
    for pid in pair_ids:
        for c in CONDITIONS:
            rs = by_pair[pid][c]
            per[c]["metric"].append(np.mean([r["metric"] for r in rs]))
            # pair correct if both items match
            per[c]["pair_ok"].append(float(all(r["canonical_match"] for r in rs)))
            per[c]["nll"].append(np.mean([r["nll"] for r in rs]))
            per[c]["identical"].append(float(rs[0]["pred_norm"] == rs[1]["pred_norm"]))
    arr = {c: {k: np.array(v) for k, v in per[c].items()} for c in CONDITIONS}

    def block(mask_pairs):
        # estimates and bootstrap intervals
        subset = np.where(mask_pairs)[0]
        idx = boot_idx(len(pair_ids), rng, subset)
        out = {}
        for c in CONDITIONS:
            rows = [r for pid in np.array(pair_ids)[subset] for r in by_pair[pid][c]]
            out[c] = {
                "accuracy": float(arr[c]["metric"][subset].mean()),
                "accuracy_ci": ci(arr[c]["metric"], idx),
                "pair_accuracy": float(arr[c]["pair_ok"][subset].mean()),
                "pair_accuracy_ci": ci(arr[c]["pair_ok"], idx),
                "identical_answer_rate": float(arr[c]["identical"][subset].mean()),
                "empty_answer_rate": float(np.mean([r["empty"] for r in rows])),
                "reference_nll": float(arr[c]["nll"][subset].mean()),
                "reference_nll_ci": ci(arr[c]["nll"], idx),
                "seconds_per_answer": float(np.mean([r["seconds"] for r in rows])),
                "n_items": len(rows),
            }
        # advantages vs complementary image
        # positive favors correct image
        acc_adv = arr["correct"]["metric"] - arr["complementary"]["metric"]
        nll_adv = arr["complementary"]["nll"] - arr["correct"]["nll"]
        pair_adv = arr["correct"]["pair_ok"] - arr["complementary"]["pair_ok"]
        out["advantage"] = {
            "accuracy_correct_minus_complementary": float(acc_adv[subset].mean()),
            "accuracy_advantage_ci": ci(acc_adv, idx),
            "pair_accuracy_advantage": float(pair_adv[subset].mean()),
            "pair_accuracy_advantage_ci": ci(pair_adv, idx),
            "nll_complementary_minus_correct": float(nll_adv[subset].mean()),
            "nll_advantage_ci": ci(nll_adv, idx),
        }
        # complementary vs partner answer
        comp = [r for pid in np.array(pair_ids)[subset] for r in by_pair[pid]["complementary"]]
        out["complementary_follows_partner_rate"] = float(
            np.mean([r["pred_norm"] == r["partner_answer_norm"] for r in comp]))
        # unsupported matches neither answer
        out["complementary_unsupported_rate"] = float(
            np.mean([r["pred_norm"] not in (r["answer_norm"], r["partner_answer_norm"]) for r in comp]))
        return out

    result = {"overall": block(np.ones(len(pair_ids), dtype=bool)), "by_type": {}}
    for t in types:
        result["by_type"][t] = block(pair_type == t)
    return result


# main

def main():
    # command line
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT.parent.parent / "artifacts" / "task2" / "main"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    run = Path(args.run)
    ev = run / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(SEED)

    # load stage 3 weights
    vlm = VLM(device)
    ck = torch.load(run / "stage3_vqa.pt", map_location=device, weights_only=False)
    vlm.scale = ck["scale"]
    vlm.connector.load_state_dict(ck["connector"])
    vlm.core.load_state_dict(ck["lora"], strict=False)
    assert len(ck["lora"]) == sum(1 for n in vlm.core.state_dict() if "lora_" in n), "LoRA keys missing"
    # eval mode
    vlm.eval()
    feats = Features()

    # suites and mismatch map
    suites = {"vqa_test": read_jsonl("vqa_test.jsonl"), "grounding_test": read_jsonl("grounding_test.jsonl")}
    mismatch = json.load(open(ROOT / "manifests" / "mismatch_test.json"))

    # generate and score
    results = {}
    all_items = []
    for suite, pairs in suites.items():
        items = make_items(suite, pairs, mismatch)
        # batches per condition
        for cond in CONDITIONS:
            group = [r for r in items if r["cond"] == cond]
            for i in range(0, len(group), BATCH):
                recs = group[i:i + BATCH]
                texts, nll, secs = run_batch(vlm, feats, recs, with_image=(cond != "no_image"))
                for r, t, n in zip(recs, texts, nll):
                    r["pred_text"] = t
                    r["nll"] = n
                    r["seconds"] = secs
        score_items(suite, items)
        results[suite] = summarize(suite, pairs, items, rng)
        all_items += items
        print(f"{suite}: {len(items)} items scored", flush=True)

    # write outputs
    with open(ev / "predictions.jsonl", "w") as f:
        for r in all_items:
            f.write(json.dumps({k: r[k] for k in ["suite", "pair_id", "k", "type", "question", "cond",
                                                  "image_id", "answer", "answers_all", "answer_norm", "pred", "pred_norm",
                                                  "canonical_match", "metric", "nll", "seconds"]}) + "\n")
    (ev / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({s: results[s]["overall"]["correct"]["accuracy"] for s in results}), flush=True)


if __name__ == "__main__":
    main()
