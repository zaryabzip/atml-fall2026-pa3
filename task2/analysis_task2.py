# task 2 extra analysis: agreement between reference likelihood and generated answers, per question,
# and a latex version of the saved qualitative pairs. reads the saved predictions only
# python task2/analysis_task2.py --run results/task2/main --tables report/tables

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabkit import TODO_CAPTION, table  # noqa: E402

SUITE = {"grounding_test": "Grounding", "vqa_test": "VQAv2"}


def tex(s, n=48):
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "^": r"\^{}",
           "_": r"\_", "%": r"\%", "~": r"\~{}"}
    s = " ".join(str(s).replace("**", "").split())
    s = s if len(s) <= n else s[:n - 3] + "..."
    return "".join(rep.get(c, c) for c in s)


def consistency(preds):
    # one question = (suite, pair, item). likelihood view: the reference answer is less surprising under the correct
    # image than under the partner image. generation view: correct under the correct image and a different answer
    # under the partner image
    by = defaultdict(dict)
    for p in preds:
        by[(p["suite"], p["pair_id"], p["k"])][p["cond"]] = p
    out = {}
    for suite in SUITE:
        cells = {"both": 0, "likelihood_only": 0, "generation_only": 0, "neither": 0}
        gen_correct, nll_pref_given_correct = 0, 0
        for (s, _, _), c in by.items():
            if s != suite:
                continue
            nll_pref = c["correct"]["nll"] < c["complementary"]["nll"]
            gen = c["correct"]["canonical_match"] and c["correct"]["pred_norm"] != c["complementary"]["pred_norm"]
            cells["both" if nll_pref and gen else "likelihood_only" if nll_pref else "generation_only" if gen else "neither"] += 1
            if c["correct"]["canonical_match"]:
                gen_correct += 1
                nll_pref_given_correct += nll_pref
        n = sum(cells.values())
        out[suite] = {**cells, "questions": n, "agreement": (cells["both"] + cells["neither"]) / n,
                      "likelihood_prefers_correct": (cells["both"] + cells["likelihood_only"]) / n,
                      "generation_grounded": (cells["both"] + cells["generation_only"]) / n,
                      "correct_under_correct_image": gen_correct,
                      "likelihood_prefers_correct_given_correct": nll_pref_given_correct / max(gen_correct, 1)}
    return out


def cut(text):
    # long answers that stop mid-sentence were cut by the generation cap
    t = text.strip()
    return r" \ldots" if len(t.split()) > 3 and not t.endswith((".", "?", "!")) else ""


def qualitative_rows(md):
    # parse the saved qualitative.md: one section per pair, one row per item
    rows, head, q, seen, skip = [], None, None, set(), False
    for line in open(md):
        m = re.match(r"## (\S+) (\S+) \((.+)\)", line)
        if m:
            head = (SUITE.get(m.group(1), m.group(1)), m.group(3))
            skip = head in seen  # one pair per suite and answer type
            seen.add(head)
            continue
        if skip:
            continue
        if line.startswith("question:"):
            q = line.split(":", 1)[1].strip()
            continue
        if line.startswith("| A |") or line.startswith("| B |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append([f"{head[0]}, {head[1]}", tex(q, 200) if cells[0] == "A" else "", cells[0]]
                        + [tex(c, 36) for c in cells[1:-1]] + [tex(cells[-1], 200) + cut(cells[-1])])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tables", required=True)
    args = ap.parse_args()
    run, tab = Path(args.run), Path(args.tables)
    preds = [json.loads(l) for l in open(run / "eval" / "predictions.jsonl")]
    res = consistency(preds)
    (run / "report" / "consistency.json").write_text(json.dumps(res, indent=1))
    rows = []
    for s, r in res.items():
        rows.append([SUITE[s], r["questions"], r["both"], r["likelihood_only"], r["generation_only"], r["neither"],
                     f"{100 * r['agreement']:.1f}", f"{100 * r['likelihood_prefers_correct_given_correct']:.1f}"])
    tab.joinpath("t2_consistency.tex").write_text(table(
        TODO_CAPTION, "tab:t2_consistency",
        ["Suite", "Questions", "Both", r"Likelihood\\only", r"Generation\\only", "Neither", r"Agreement\\(\%)",
         r"Likelihood prefers\\correct, answer\\right (\%)"], rows, "lrrrrrrr",
        top=[("", 2), ("Image use shown by", 4), ("", 2)]))
    tab.joinpath("t2_qualitative.tex").write_text(table(
        TODO_CAPTION, "tab:t2_qualitative", ["Question", "Item", "Reference", "Correct", "Compl.", "Mism.", "No image"],
        qualitative_rows(run / "report" / "qualitative.md"), r"p{3.0cm}cllllp{3.8cm}", sections=True, size=r"\small"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
