# task 2 data manifests

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vqa_contractions import CONTRACTIONS

# settings

SEED = 6304
ROOT = Path(__file__).resolve().parent.parent / "data" / "task2"
COCO = ROOT / "coco"
VQA = ROOT / "vqa"
OUT = ROOT / "manifests"

CAPTIONS_PER_TRAIN_IMAGE = 3
N_CAPTION_TRAIN_IMAGES = 6000
N_CAPTION_DEV_IMAGES = 96
N_PRESENCE_POOL = 4000
N_COUNTING_POOL = 2000
GROUNDING_DEV = {"presence": 12, "counting": 12}  # 24 random pairs
GROUNDING_TEST = {"presence": 32, "counting": 32}
VQA_TRAIN = {"yes/no": 640, "number": 256, "other": 640}
VQA_DEV = {"yes/no": 12, "number": 8, "other": 12}
VQA_TEST = {"yes/no": 36, "number": 24, "other": 36}

# plural rules

IRREGULAR_PLURALS = {"person": "people", "knife": "knives", "mouse": "mice"}
UNCHANGED_PLURALS = {"sheep", "scissors", "skis", "broccoli"}

# answer normalization

MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
              "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@",
         "`", ",", "?", "!"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def normalize_answer(text):
    text = text.replace("\n", " ").replace("\t", " ").strip()
    out = text
    # punctuation step
    for p in PUNCT:
        if (p + " " in text or " " + p in text) or COMMA_STRIP.search(text):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = PERIOD_STRIP.sub("", out)
    # number words articles contractions
    words = [MANUAL_MAP.get(w, w) for w in out.lower().split() if w not in ARTICLES]
    words = [CONTRACTIONS.get(w, w) for w in words]
    return " ".join(words)



def make_mismatch(pairs, rng):
    # mismatch map for one suite
    mapping = {}
    for p in pairs:
        own = {x["image_id"] for x in p["items"]}
        pool = sorted({x["image_id"] for q in pairs if q["pair_id"] != p["pair_id"]
                       for x in q["items"]} - own)
        for k in range(len(p["items"])):
            mapping[f"{p['pair_id']}|{k}"] = int(pool[rng.integers(len(pool))])
    return mapping


# helpers

def consensus(answers):
    # at least 8 of 10 agree
    norm = [normalize_answer(a["answer"]) for a in answers]
    best, count = Counter(norm).most_common(1)[0]
    if count < 8:
        return None
    raw = next(a["answer"] for a, n in zip(answers, norm) if n == best)
    return best, raw


def plural(name):
    if name in UNCHANGED_PLURALS:
        return name
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


def article(name):
    return "an" if name[0] in "aeiou" else "a"


# coco loading

def load_coco(split):
    caps = json.load(open(COCO / "annotations" / f"captions_{split}2014.json"))
    inst = json.load(open(COCO / "annotations" / f"instances_{split}2014.json"))
    files = {im["id"]: f"{split}2014/{im['file_name']}" for im in caps["images"]}
    captions = defaultdict(list)
    for a in sorted(caps["annotations"], key=lambda a: a["id"]):
        captions[a["image_id"]].append(a["caption"].strip())
    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    objs = defaultdict(lambda: defaultdict(int))
    for a in inst["annotations"]:
        objs[a["image_id"]][cat_name[a["category_id"]]] += 1
    return files, captions, objs


# grounding pairs

def make_presence(n, pool, objs, rng, used):
    # presence pairs
    by_cat = defaultdict(list)
    for img in pool:
        for c in objs.get(img, {}):
            by_cat[c].append(img)
    cats = sorted(by_cat)
    rows = []
    while len(rows) < n:
        c = cats[rng.integers(len(cats))]
        pos = by_cat[c][rng.integers(len(by_cat[c]))]
        neg = pool[rng.integers(len(pool))]
        # resample on conflict
        if c in objs.get(neg, {}) or pos == neg or pos in used or neg in used:
            continue
        used.update([pos, neg])
        rows.append({"type": "presence", "question": f"Does this image contain {article(c)} {c}?",
                     "items": [(pos, "yes"), (neg, "no")]})
    return rows


def make_counting(n, pool, objs, rng, used):
    # counting pairs
    by_cat = defaultdict(lambda: defaultdict(list))
    for img in pool:
        for c, k in objs.get(img, {}).items():
            if 1 <= k <= 4:
                by_cat[c][k].append(img)
    cats = sorted(c for c in by_cat if len(by_cat[c]) >= 2)
    rows = []
    while len(rows) < n:
        c = cats[rng.integers(len(cats))]
        ks = sorted(by_cat[c])
        k1, k2 = (int(k) for k in rng.choice(ks, size=2, replace=False))
        i1 = by_cat[c][k1][rng.integers(len(by_cat[c][k1]))]
        i2 = by_cat[c][k2][rng.integers(len(by_cat[c][k2]))]
        if i1 in used or i2 in used:
            continue
        used.update([i1, i2])
        rows.append({"type": "counting", "question": f"How many {plural(c)} can you see?",
                     "items": [(i1, str(k1)), (i2, str(k2))]})
    return rows


def finish_grounding(rows, prefix, files):
    # reshape to vqa layout
    out = []
    for i, r in enumerate(rows):
        items = [{"image_id": img, "file": files[img], "answer": ans, "answer_norm": ans,
                  "answers_all": [ans]} for img, ans in r["items"]]
        out.append({"pair_id": f"{prefix}_{i:05d}", "type": r["type"], "question": r["question"],
                    "items": items})
    return out


# vqav2 selection

def load_vqa(split):
    # split files
    qs = {q["question_id"]: q for q in
          json.load(open(VQA / f"v2_OpenEnded_mscoco_{split}2014_questions.json"))["questions"]}
    an = {a["question_id"]: a for a in
          json.load(open(VQA / f"v2_mscoco_{split}2014_annotations.json"))["annotations"]}
    comp = json.load(open(VQA / f"v2_mscoco_{split}2014_complementary_pairs.json"))
    return qs, an, comp


def vqa_candidates(comp, qs, an):
    # keep pairs with differing consensus
    cands = defaultdict(list)
    for q1, q2 in comp:
        if q1 not in an or q2 not in an:
            continue
        c1 = consensus(an[q1]["answers"])
        c2 = consensus(an[q2]["answers"])
        if c1 is None or c2 is None or c1[0] == c2[0]:
            continue
        if an[q1]["answer_type"] != an[q2]["answer_type"]:
            continue
        cands[an[q1]["answer_type"]].append((q1, q2, c1, c2))
    return cands


def select_vqa(cands, quotas, qs, an, files, split, rng, used, check_used):
    # seeded draw per answer type
    rows = []
    for atype, n in quotas.items():
        order = rng.permutation(len(cands[atype]))
        taken = 0
        for idx in order:
            if taken == n:
                break
            q1, q2, c1, c2 = cands[atype][idx]
            imgs = [qs[q1]["image_id"], qs[q2]["image_id"]]
            if check_used and any(i in used for i in imgs):
                continue
            used.update(imgs)
            items = []
            for q, c, img in ((q1, c1, imgs[0]), (q2, c2, imgs[1])):
                # keep all ten answers
                items.append({"image_id": img, "file": files[img], "answer": c[1],
                              "answer_norm": c[0],
                              "answers_all": [a["answer"] for a in an[q]["answers"]]})
            rows.append({"pair_id": f"vqa_{split}_{q1}_{q2}", "type": atype,
                         "question": qs[q1]["question"], "items": items})
            taken += 1
        if taken < n:
            raise SystemExit(f"{split} {atype}: only {taken} of {n} qualifying pairs")
    return rows


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {path.name}")


# main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    files_tr, caps_tr, objs_tr = load_coco("train")
    files_va, caps_va, objs_va = load_coco("val")
    used_val = set()
    used_train = set()

    # step 1 vqa val pairs
    qs_va, an_va, comp_va = load_vqa("val")
    cands_va = vqa_candidates(comp_va, qs_va, an_va)
    dev_vqa = select_vqa(cands_va, VQA_DEV, qs_va, an_va, files_va, "dev", rng, used_val, True)
    test_vqa = select_vqa(cands_va, VQA_TEST, qs_va, an_va, files_va, "test", rng, used_val, True)

    # step 2 grounding val pairs
    val_pool = sorted(i for i in files_va if i not in used_val)
    dev_ground = make_presence(GROUNDING_DEV["presence"], val_pool, objs_va, rng, used_val)
    dev_ground += make_counting(GROUNDING_DEV["counting"], val_pool, objs_va, rng, used_val)
    test_ground = make_presence(GROUNDING_TEST["presence"], val_pool, objs_va, rng, used_val)
    test_ground += make_counting(GROUNDING_TEST["counting"], val_pool, objs_va, rng, used_val)

    # step 3 caption dev images
    val_caption_pool = sorted(i for i in files_va if caps_va.get(i) and i not in used_val)
    dev_cap_imgs = [int(i) for i in rng.choice(val_caption_pool, N_CAPTION_DEV_IMAGES, replace=False)]
    used_val.update(dev_cap_imgs)
    caption_dev = [{"image_id": i, "file": files_va[i], "caption": caps_va[i][0]} for i in dev_cap_imgs]

    # step 4 train pools
    train_pool = sorted(files_tr)
    cap_eligible = sorted(i for i in train_pool if len(caps_tr.get(i, [])) >= CAPTIONS_PER_TRAIN_IMAGE)
    cap_imgs = [int(i) for i in rng.choice(cap_eligible, N_CAPTION_TRAIN_IMAGES, replace=False)]
    caption_train = [{"image_id": i, "file": files_tr[i], "caption": caps_tr[i][k], "caption_idx": k}
                     for i in cap_imgs for k in range(CAPTIONS_PER_TRAIN_IMAGE)]
    presence_pool = make_presence(N_PRESENCE_POOL, train_pool, objs_tr, rng, used_train)
    counting_pool = make_counting(N_COUNTING_POOL, train_pool, objs_tr, rng, used_train)

    # step 5 vqa train pairs
    qs_tr, an_tr, comp_tr = load_vqa("train")
    cands_tr = vqa_candidates(comp_tr, qs_tr, an_tr)
    vqa_train = select_vqa(cands_tr, VQA_TRAIN, qs_tr, an_tr, files_tr, "train", rng, set(), False)

    # step 6 mismatch map
    test_ground_pairs = [{"pair_id": f"gtest_{i:05d}", "items": [{"image_id": img} for img, _ in r["items"]]}
                         for i, r in enumerate(test_ground)]
    mismatch = {}
    mismatch.update(make_mismatch(test_vqa, rng))
    mismatch.update(make_mismatch(test_ground_pairs, rng))
    json.dump(mismatch, open(OUT / "mismatch_test.json", "w"))

    # save manifests
    write_jsonl(OUT / "captions_train.jsonl", caption_train)
    write_jsonl(OUT / "captions_dev.jsonl", caption_dev)
    write_jsonl(OUT / "presence_pool_train.jsonl", finish_grounding(presence_pool, "pres", files_tr))
    write_jsonl(OUT / "counting_pool_train.jsonl", finish_grounding(counting_pool, "count", files_tr))
    write_jsonl(OUT / "grounding_dev.jsonl", finish_grounding(dev_ground, "gdev", files_va))
    write_jsonl(OUT / "grounding_test.jsonl", finish_grounding(test_ground, "gtest", files_va))
    write_jsonl(OUT / "vqa_train.jsonl", vqa_train)
    write_jsonl(OUT / "vqa_dev.jsonl", dev_vqa)
    write_jsonl(OUT / "vqa_test.jsonl", test_vqa)

    # image sets
    image_sets = {
        "caption_train": cap_imgs,
        "caption_dev": dev_cap_imgs,
        "pool_train": sorted({i for r in presence_pool + counting_pool for i, _ in r["items"]}),
        "vqa_train": sorted({it["image_id"] for r in vqa_train for it in r["items"]}),
        "vqa_dev": sorted({it["image_id"] for r in dev_vqa for it in r["items"]}),
        "vqa_test": sorted({it["image_id"] for r in test_vqa for it in r["items"]}),
        "grounding_dev": sorted({i for r in dev_ground for i, _ in r["items"]}),
        "grounding_test": sorted({i for r in test_ground for i, _ in r["items"]}),
    }
    json.dump(image_sets, open(OUT / "image_sets.json", "w"))

    # leakage checks
    eval_sets = ["caption_dev", "vqa_dev", "vqa_test", "grounding_dev", "grounding_test"]
    for a in eval_sets:
        assert not set(image_sets[a]) & set(image_sets["caption_train"]), f"{a} overlaps caption_train"
        assert not set(image_sets[a]) & set(image_sets["pool_train"]), f"{a} overlaps pool_train"
        assert not set(image_sets[a]) & set(image_sets["vqa_train"]), f"{a} overlaps vqa_train"
    for i, a in enumerate(eval_sets):
        for b in eval_sets[i + 1:]:
            assert not set(image_sets[a]) & set(image_sets[b]), f"{a} overlaps {b}"
    print("leakage checks passed")


if __name__ == "__main__":
    main()
