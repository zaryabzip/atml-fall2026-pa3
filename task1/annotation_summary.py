# table of the 42-response annotation subset: labels per condition, written to report/tables/t1_annotation.tex
# python task1/annotation_summary.py
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabkit import table

NAMES = {"baseline": "Baseline", "full_mid": "Full FT, mid", "full_final": "Full FT, final", "l2sp_final": "L2-SP, final",
         "lora4_final": "LoRA-4, final", "lora16_final": "LoRA-16, final", "oplora16_final": "OPLoRA-16, final"}
CAPTION = ("Labels for the 42-response annotation subset, the first and last non-math questions in all three formats, "
           "six responses per condition. Format compliance is out of the 4 JSON and template responses. "
           "Coherence is the number of responses that were partly or not relevant, and Math framing the number with an unneeded "
           "``The answer is'' line. Entries are numbers of responses, and none were inappropriate.")
rows = list(csv.DictReader(open(ROOT / "results/task1/qualitative/annotation_subset.csv")))
out = []
for v, name in NAMES.items():
    rs = [r for r in rows if r["variant"] == v]
    structured = [r for r in rs if r["format"] != "plain"]
    out.append([name,
                f"{sum(r['format_compliance'] == 'yes' for r in structured)}/{len(structured)}",
                f"{sum(r['coherence'] in ('partial', 'poor') for r in rs)}/{len(rs)}",
                f"{sum(r['repetition'] in ('some', 'heavy') for r in rs)}/{len(rs)}",
                f"{sum(r['unnecessary_math'] == 'yes' for r in rs)}/{len(rs)}",
                f"{sum(r['truncated'] == 'True' for r in rs)}/{len(rs)}"])
(ROOT / "report/tables/t1_annotation.tex").write_text(table(
    CAPTION, "tab:t1_annotation",
    ["Condition", r"Format\\compliant", r"Coherence\\problems", "Repetition", r"Math\\framing", "Truncated"],
    out, "lrrrrr"))
print("wrote report/tables/t1_annotation.tex")
