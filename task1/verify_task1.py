# checks the task 1 training data and the saved models against the spec. cpu only.
# python task1/verify_task1.py --artifacts artifacts/task1 --out results/task1/verification.json

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

QWEN = "Qwen/Qwen3-0.6B"
LAYERS = (0, 13, 27)
RUNS_LORA = {"lora4": (4, 8), "lora16": (16, 32), "oplora16": (16, 32)}


def norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def base_state():
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download(QWEN, local_files_only=True))
    return load_file(str(p / "model.safetensors")), json.load(open(p / "config.json"))


def check_data(lora_root):
    # recompute the first 40,000 eligible metamathqa rows with the spec rules and compare with the saved row ids
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    gsm_q = {norm(q) for q in load_dataset("openai/gsm8k", "main", split="test")["question"]}
    path = hf_hub_download("meta-math/MetaMathQA", "MetaMathQA-395K.json", repo_type="dataset", local_files_only=True)
    rows = json.load(open(path))
    kept, excluded = [], {"empty": 0, "gsm8k_match": 0, "over_2048_tokens": 0}
    for i, r in enumerate(rows):
        q, a = r["query"], r["response"]
        if not q.strip() or not a.strip():
            excluded["empty"] += 1
            continue
        if norm(q) in gsm_q or norm(r.get("original_question", "")) in gsm_q:
            excluded["gsm8k_match"] += 1
            continue
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        n = len(tok(prompt, add_special_tokens=False).input_ids) + len(tok(a + "<|im_end|>", add_special_tokens=False).input_ids)
        if n > 2048:
            excluded["over_2048_tokens"] += 1
            continue
        kept.append(i)
        if len(kept) == 40000:
            break
    saved = json.load(open(lora_root / "train_examples.json"))["row_ids"]
    return {"recomputed_rows": len(kept), "matches_saved_row_ids": kept == saved, "rows_scanned": i + 1,
            "excluded_before_cutoff": excluded, "source_rows": len(rows)}


def check_order(full_root, lora_root):
    # update 1 runs at lr 0, so update 2 also sees the unchanged model. matching losses at updates 1 and 2
    # across all five runs mean they saw the same first two batches
    losses = {}
    for n in ["full", "l2sp"]:
        rows = {json.loads(l)["update"]: json.loads(l) for l in open(full_root / n / "train_log.jsonl")}
        losses[n] = [rows[1]["sft_loss"], rows[2]["sft_loss"]]
    for n in RUNS_LORA:
        rows = {r["update"]: r for r in json.load(open(lora_root / n / "train_log.json"))}
        losses[n] = [rows[1]["sft_loss"], rows[2]["sft_loss"]]
    u1 = [v[0] for v in losses.values()]
    u2 = [v[1] for v in losses.values()]
    return {"losses_update_1_2": losses, "spread_update_1": max(u1) - min(u1), "spread_update_2": max(u2) - min(u2)}


def check_snapshot(path, base_sd, tied, logged_disp):
    sd = load_file(str(path / "model.safetensors"))
    fp32 = torch.load(path / "down_proj_fp32.pt", weights_only=False)
    down_diff = max(float((sd[f"model.layers.{L}.mlp.down_proj.weight"].float() - fp32[L].half().float()).abs().max())
                    for L in LAYERS)
    keys = [k for k in base_sd if not (tied and k == "lm_head.weight")]
    sq = 0.0
    for k in keys:
        sq += float(((sd[k].float() - base_sd[k].float()) ** 2).sum())
    return {"fp16_matches_fp32_down_proj_max_diff": down_diff, "lm_head_saved": "lm_head.weight" in sd,
            "missing_keys": [k for k in keys if k not in sd][:5], "displacement_from_fp16_snapshot": sq ** 0.5,
            "displacement_logged_fp32": logged_disp, "dtype": str(next(iter(sd.values())).dtype)}


def check_adapters(name, lora_root):
    rank, alpha = RUNS_LORA[name]
    data = torch.load(lora_root / name / "snapshot_1250" / "adapters.pt", weights_only=False)
    mods = data["modules"]
    da, db, n_params, scales = 0.0, 0.0, 0, set()
    for mname, m in mods.items():
        A, B = m["A"].float(), m["B"].float()
        if name.startswith("oplora"):
            Uk, Vk = data["bases"][mname]["Uk"].float(), data["bases"][mname]["Vk"].float()
            A_exp, B_exp = A - (A @ Vk) @ Vk.T, B - Uk @ (Uk.T @ B)
        else:
            A_exp, B_exp = A, B
        da = max(da, float((m["A_eff"] - A_exp).abs().max()))
        db = max(db, float((m["B_eff"] - B_exp).abs().max()))
        n_params += A.numel() + B.numel()
        scales.add(round(m["scale"], 6))
        assert A.shape[0] == rank and B.shape[1] == rank, mname
    targets = sorted({k.split(".")[-1] for k in mods})
    return {"modules": len(mods), "targets": targets, "trainable_params": n_params, "scales": sorted(scales),
            "expected_scale": alpha / rank, "A_eff_vs_recomputed_max_diff": da, "B_eff_vs_recomputed_max_diff": db,
            "meta": {k: data["meta"][k] for k in ("method", "rank", "alpha", "protect_k") if k in data["meta"]},
            "checks": data.get("checks", {})}


def check_bases(lora_root, base_sd):
    # the oplora bases should span the top 128 singular directions of the fp16 starting weight
    data = torch.load(lora_root / "oplora16" / "snapshot_1250" / "adapters.pt", weights_only=False)
    out = {}
    for mname in ["model.layers.0.self_attn.q_proj", "model.layers.13.mlp.down_proj", "model.layers.27.mlp.up_proj"]:
        W0 = base_sd[mname + ".weight"].half().float()
        U, _, Vh = torch.linalg.svd(W0, full_matrices=False)
        Uk_ref, Vk_ref = U[:, :128], Vh[:128].T
        Uk, Vk = data["bases"][mname]["Uk"].float(), data["bases"][mname]["Vk"].float()
        pu = float((Uk @ Uk.T - Uk_ref @ Uk_ref.T).norm() / (Uk_ref @ Uk_ref.T).norm())
        pv = float((Vk @ Vk.T - Vk_ref @ Vk_ref.T).norm() / (Vk_ref @ Vk_ref.T).norm())
        ortho = float((Uk.T @ Uk - torch.eye(128)).abs().max())
        out[mname] = {"output_projector_rel_diff": pu, "input_projector_rel_diff": pv, "Uk_orthonormal_max_err": ortho}
    return out


def check_merged(name, lora_root, base_sd):
    # merged = fp16(fp32 starting weight + scale * B_eff A_eff) on every target, starting weight elsewhere
    merged = load_file(str(lora_root / "merged" / name / "model.safetensors"))
    mods = torch.load(lora_root / name / "snapshot_1250" / "adapters.pt", weights_only=False)["modules"]
    t_diff, o_diff, n_t, n_o = 0.0, 0.0, 0, 0
    for key, w in merged.items():
        if key not in base_sd:
            continue
        mname = key[: -len(".weight")] if key.endswith(".weight") else None
        b = base_sd[key].float()
        if mname in mods:
            m = mods[mname]
            exp = (b + m["scale"] * (m["B_eff"].float() @ m["A_eff"].float())).half().float()
            t_diff = max(t_diff, float((w.float() - exp).abs().max()))
            n_t += 1
        else:
            o_diff = max(o_diff, float((w.float() - b.half().float()).abs().max()))
            n_o += 1
    return {"target_tensors": n_t, "other_tensors": n_o, "target_max_abs_diff": t_diff, "other_max_abs_diff": o_diff,
            "dtype": str(next(iter(merged.values())).dtype)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_data", action="store_true")
    args = ap.parse_args()
    A = Path(args.artifacts)
    full_root, lora_root = A / "full_l2sp_kaggle", A / "lora_kaggle" / "task1"
    base_sd, cfg = base_state()
    tied = bool(cfg.get("tie_word_embeddings"))
    report = {"base_unique_params": sum(t.numel() for k, t in base_sd.items() if not (tied and k == "lm_head.weight")),
              "tie_word_embeddings": tied}
    report["order"] = check_order(full_root, lora_root)
    full_sum = json.load(open(full_root / "full" / "summary.json"))
    l2sp_sum = json.load(open(full_root / "l2sp" / "summary.json"))
    full_rows = {json.loads(l)["update"]: json.loads(l) for l in open(full_root / "full" / "train_log.jsonl")}
    report["snapshots"] = {
        "full_625": check_snapshot(full_root / "full" / "snapshot_625", base_sd, tied, full_rows[625]["displacement"]),
        "full_1250": check_snapshot(full_root / "full" / "snapshot_1250", base_sd, tied, full_sum["final_displacement"]),
        "l2sp_1250": check_snapshot(full_root / "l2sp" / "snapshot_1250", base_sd, tied, l2sp_sum["final_displacement"]),
    }
    report["summaries_trainable_params"] = {"full": full_sum["trainable_params"], "l2sp": l2sp_sum["trainable_params"]}
    report["adapters"] = {n: check_adapters(n, lora_root) for n in RUNS_LORA}
    report["oplora_bases"] = check_bases(lora_root, base_sd)
    report["merged"] = {n: check_merged(n, lora_root, base_sd) for n in RUNS_LORA}
    if not args.skip_data:
        report["data"] = check_data(lora_root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
