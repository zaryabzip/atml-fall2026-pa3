# answer normalization, grouping and grading for task 3.
# math uses math-verify equivalence. mathvista uses the official answer-type, choice and precision rules
# (lupantech/MathVista evaluation/calculate_score.py, normalize_extracted_answer and get_most_similar).
# missing or empty extractions form one NO_ANSWER group, always graded incorrect.

import re

from math_verify import parse, verify

NO_ANSWER = "NO_ANSWER"


def missing(x):
    return x is None or (isinstance(x, str) and not x.strip())


# math

_PARSED = {}


def mparse(s):
    if s not in _PARSED:
        _PARSED[s] = parse("$" + s + "$")
    return _PARSED[s]


def math_equiv(a, b):
    pa, pb = mparse(a), mparse(b)
    if not pa or not pb:
        return False
    return bool(verify(pa, pb) or verify(pb, pa))


def group_math(answers):
    # exact strings first, then greedy merge in order of first appearance (rollout id order)
    reps, key_of, labels = [], {}, []
    for a in answers:
        if missing(a):
            labels.append(NO_ANSWER)
            continue
        s = a.strip()
        if s not in key_of:
            key_of[s] = next((r for r in reps if math_equiv(r, s)), None) or s
            if key_of[s] == s:
                reps.append(s)
        labels.append(key_of[s])
    return labels


def grade_math(rep, gold):
    if rep == NO_ANSWER:
        return False
    pg, pr = mparse(gold), mparse(rep)
    return bool(pg and pr and verify(pg, pr))


# mathvista

def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def most_similar(pred, choices):
    d = [levenshtein(pred, c) for c in choices]
    return choices[d.index(min(d))]  # first minimum wins, as in the official code


def mv_choice_letter(extraction, choices):
    # official rule: "(X)" gives a letter, a bare letter indexes the choices, otherwise the nearest choice
    e = extraction.strip()
    letter = re.findall(r"\(([a-zA-Z])\)", e)
    if letter:
        e = letter[0].upper()
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    choice = choices[letters.index(e)] if e in letters else most_similar(e, choices)
    return letters[choices.index(choice)]


def mv_number(extraction, answer_type, precision):
    try:
        if answer_type == "integer":
            return str(int(float(extraction)))
        if answer_type == "float":
            return str(round(float(extraction), int(precision)))
    except (ValueError, TypeError, OverflowError):
        return None
    return None


def mv_clean(s):
    # formatting only, applied when plain float() fails: units in text blocks, degree and percent signs,
    # latex spacing, thousands commas, simple fractions and scientific notation. units are never converted.
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(?:text|mathrm)\{[^{}]*\}", " ", s)
    s = re.sub(r"\^\{?\\circ\}?|\\circ|°|\\%|%", " ", s)
    s = re.sub(r"\\[,;:! ]", " ", s).strip()
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    m = re.fullmatch(r"(-?)\\[dt]?frac\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}", s)
    if m and float(m.group(3)) != 0:
        return repr((-1 if m.group(1) else 1) * float(m.group(2)) / float(m.group(3)))
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?", s)
    if m:
        return repr(float(m.group(1)) * 10 ** int(m.group(2)))
    return s


SCORER_PATHS = {"raw": 0, "cleaned": 0, "unparsed": 0}


def normalize_mathvista(extraction, meta, clean=mv_clean):
    # returns the answer group key. unparseable numbers keep their own key and grade incorrect
    if missing(extraction):
        return NO_ANSWER
    if meta["question_type"] == "multi_choice":
        return "CHOICE:" + mv_choice_letter(str(extraction), meta["choices"])
    at = meta["answer_type"]
    if at == "list":
        return "LIST:" + str(extraction).strip()
    raw = str(extraction).strip()
    value = mv_number(raw, at, meta.get("precision"))
    if value is not None:
        SCORER_PATHS["raw"] += 1
        return "NUM:" + value
    cleaned = clean(raw) if clean else raw
    value = mv_number(cleaned, at, meta.get("precision"))
    if value is not None:
        SCORER_PATHS["cleaned"] += 1
        return "NUM:" + value
    SCORER_PATHS["unparsed"] += 1
    return "UNPARSED:" + (cleaned or raw)


def gold_mathvista(gold, meta):
    if meta["question_type"] == "multi_choice":
        assert gold in meta["choices"], f"gold not among the choices: {gold!r}"  # duplicates map to the first
        letters = [chr(ord("A") + i) for i in range(len(meta["choices"]))]
        return "CHOICE:" + letters[meta["choices"].index(gold)]
    if meta["answer_type"] == "list":
        return "LIST:" + str(gold).strip()
    value = mv_number(str(gold), meta["answer_type"], meta.get("precision"))
    assert value is not None, f"gold not numeric: {gold!r}"
    return "NUM:" + value


def grade_mathvista(key, gold_key):
    return key != NO_ANSWER and not key.startswith("UNPARSED:") and key == gold_key
