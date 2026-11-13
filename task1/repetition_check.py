# task 1 rq5: what the repeated 4-grams are, from the saved qualitative responses (no model is loaded)
# python task1/repetition_check.py   writes results/task1/analysis/repetition.csv and report/tables/t1_repetition.tex
import collections
import csv
import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from tabkit import TODO_CAPTION, num, table  # noqa: E402

VARIANTS = [("baseline", "Baseline"), ("full_final", "Full FT"), ("l2sp_final", "L2-SP"), ("lora4_final", "LoRA-4"),
            ("lora16_final", "LoRA-16"), ("oplora16_final", "OPLoRA-16")]
recs = json.load(open(R / "results" / "task1" / "qualitative" / "responses.json"))


def rep(text):
    w = text.split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


rows = []
for v, name in VARIANTS:
    rs = [r for r in recs if r["variant"] == v]
    shares = [rep(r["response"]) for r in rs]
    js = [r for r in rs if r["format"] == "json"]
    no_obj = sum(1 for r in js if not re.search(r"\{.*\}", r["response"], re.S))
    rows.append({"name": name, "loops": sum(s > 0.5 for s in shares), "n": len(rs), "max_share": 100 * max(shares),
                 "json_no_object": no_obj, "json_n": len(js)})
with open(R / "results" / "task1" / "analysis" / "repetition.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
body = [[r["name"], f"{r['loops']}/{r['n']}", num(r["max_share"]), f"{r['json_no_object']}/{r['json_n']}"] for r in rows]
(R / "report" / "tables" / "t1_repetition.tex").write_text(table(
    TODO_CAPTION, "tab:t1_repetition", ["Condition", r"Answers mostly\\repeated", r"Largest repeated\\share (\%)", r"JSON answers\\without an object"],
    body, "lrrr"))
print(json.dumps(rows, indent=0))
