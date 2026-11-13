# shared latex table helpers for the report table scripts
# escaping skips math, numbers get a real minus sign, repeated leading cells are blanked with a rule between groups,
# header cells with \\ are stacked, and an optional top row groups columns

import re

TODO_CAPTION = "[caption to write]"


def esc(s):
    # escape special characters outside $...$ and leave anything already escaped alone
    parts = str(s).split("$")
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<!\\)([_%&#])", r"\\\1", parts[i])
    return "$".join(parts)


def num(x, fmt=".1f", sign=False):
    # formatted number with a typographic minus
    if x is None:
        return "--"
    s = format(float(x), ("+" if sign else "") + fmt)
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def ci(lo, hi, fmt=".1f"):
    return f"[{num(lo, fmt)}, {num(hi, fmt)}]"


def stack(h):
    # header cell over two lines when it contains \\
    return rf"\begin{{tabular}}[b]{{@{{}}c@{{}}}}{h}\end{{tabular}}" if r"\\" in h else h


def table(caption, label, header, rows, colspec, note=None, group=0, top=None, size=None, escape=True, rules_after=(),
          sections=False):
    # with sections, header, colspec and top describe the columns after the section label
    if caption == TODO_CAPTION:  # captions live in captions.py, keyed by label
        try:
            from captions import CAPTIONS
            caption = CAPTIONS.get(label, caption)
        except ImportError:
            pass
    lines = [r"\begin{table}[!htb]", r"\centering"]
    if size:
        lines.append(size)
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\begin{adjustbox}{max width=\linewidth}",
              rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    if top:
        # top is a list of (text, span); a rule is drawn under every non-empty group
        cells, rules, col = [], [], 1
        for text, span in top:
            cells.append(rf"\multicolumn{{{span}}}{{c}}{{{text}}}" if span > 1 else text)
            if text:
                rules.append(rf"\cmidrule(lr){{{col}-{col + span - 1}}}")
            col += span
        lines += [" & ".join(cells) + r" \\", " ".join(rules)]
    lines += [" & ".join(stack(esc(h) if escape else h) for h in header) + r" \\", r"\midrule"]
    # wrapped p{} columns are set ragged right; rows end in \tabularnewline so that stays safe in the last column
    cols = re.findall(r"p\{[^}]*\}|[lcr]", colspec)
    ragged = {j for j, c in enumerate(cols) if c.startswith("p")}
    prev, section = [None] * group, None
    for i, r in enumerate(rows):
        r = [esc(c) if escape else str(c) for c in r]
        if sections:
            # the first cell is a section label printed as its own row across the table
            if r[0] != section:
                if i:
                    lines.append(r"\midrule")
                lines.append(rf"\multicolumn{{{len(header)}}}{{l}}{{\textit{{{r[0]}}}}} \\")
                section, prev = r[0], [None] * group
            r = r[1:]
        if group:
            if i and prev[0] is not None and r[0] != prev[0]:
                lines.append(r"\addlinespace[3pt]" if sections else r"\midrule")
            elif i and group > 1 and r[1] != prev[1]:
                lines.append(r"\addlinespace[2pt]")
            shown = list(r)
            for g in range(group):
                if r[:g + 1] == prev[:g + 1]:
                    shown[g] = ""
            prev = r[:group]
            r = shown
        r = [rf"\raggedright {c}" if j in ragged and c else c for j, c in enumerate(r)]
        lines.append(" & ".join(r) + r" \tabularnewline")
        if i in rules_after:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines += [r"\par\smallskip", rf"{{\footnotesize {note}}}"]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"
