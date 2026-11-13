# pre-training baseline for task 2. rebuilds the update-0 model from the same seed and scale,
# checks its dev numbers against dev_log update 0, then scores both test suites.
# python task2/eval_untrained.py --run artifacts/task2/main --device mps

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import eval_vlm as ev
from train_vlm import Features, SEED, dev_eval, read_jsonl
from vlm import VLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "eval_untrained"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # same order as train_vlm.main so the seeded init matches
    torch.manual_seed(SEED)
    feats = Features()
    vlm = VLM(device)
    cfg = json.loads((run / "config.json").read_text())
    vlm.scale = cfg["scale"]
    vlm.scale_rule = cfg["scale_rule"]

    # dev check against the update 0 row of the training run
    dev = {
        "caption_dev": read_jsonl("captions_dev.jsonl"),
        "grounding_dev": read_jsonl("grounding_dev.jsonl"),
        "vqa_dev": read_jsonl("vqa_dev.jsonl"),
    }
    with torch.autocast(device.type, dtype=torch.float16):
        res = dev_eval(vlm, dev, feats)
    saved = json.loads((run / "dev_log.jsonl").read_text().splitlines()[0])
    diffs = {k: abs(res[k] - saved[k]) for k in res}
    worst = max(diffs.values())
    print(json.dumps({"dev_max_abs_diff": worst, "per_key": diffs}), flush=True)
    (out / "dev_check.json").write_text(json.dumps({"dev_max_abs_diff": worst, "per_key": diffs}, indent=2))
    if worst > 1e-2:
        raise SystemExit("rebuilt model does not match dev_log update 0, stopping")

    # test suites, same scoring as eval_vlm.main
    vlm.eval()
    suites = {"vqa_test": read_jsonl("vqa_test.jsonl"), "grounding_test": read_jsonl("grounding_test.jsonl")}
    mismatch = json.load(open(ev.ROOT / "manifests" / "mismatch_test.json"))
    rng = np.random.default_rng(SEED)
    results = {}
    all_items = []
    for suite, pairs in suites.items():
        items = ev.make_items(suite, pairs, mismatch)
        for cond in ev.CONDITIONS:
            group = [r for r in items if r["cond"] == cond]
            for i in range(0, len(group), ev.BATCH):
                recs = group[i:i + ev.BATCH]
                texts, nll, secs = ev.run_batch(vlm, feats, recs, with_image=(cond != "no_image"))
                for r, t, n in zip(recs, texts, nll):
                    r["pred_text"] = t
                    r["nll"] = n
                    r["seconds"] = secs
        ev.score_items(suite, items)
        results[suite] = ev.summarize(suite, pairs, items, rng)
        all_items += items
        print(f"{suite}: {len(items)} items scored", flush=True)

    with open(out / "predictions.jsonl", "w") as f:
        for r in all_items:
            f.write(json.dumps({k: r[k] for k in ["suite", "pair_id", "k", "type", "question", "cond",
                                                  "image_id", "answer", "answers_all", "answer_norm", "pred", "pred_norm",
                                                  "canonical_match", "metric", "nll", "seconds"]}) + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({s: results[s]["overall"]["correct"]["accuracy"] for s in results}), flush=True)


if __name__ == "__main__":
    main()
