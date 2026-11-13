# cache frozen siglip features in fp16

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

# settings

ROOT = Path(__file__).resolve().parent.parent / "data" / "task2"
MANIFESTS = ROOT / "manifests"
COCO = ROOT / "coco"
FEATS = ROOT / "features"
MODEL = "google/siglip-base-patch16-224"
BATCH = 32


# image paths from manifests

def collect_files():
    files = {}
    for path in MANIFESTS.glob("*.jsonl"):
        for line in open(path):
            row = json.loads(line)
            if "items" in row:
                for it in row["items"]:
                    files[it["image_id"]] = it["file"]
            elif "image_id" in row:
                files[row["image_id"]] = row["file"]
    return files


# extraction

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    FEATS.mkdir(parents=True, exist_ok=True)
    files = collect_files()
    # sorted ids set row order
    ids = sorted(files)
    t0 = time.time()
    print(f"images to cache: {len(ids)}", flush=True)

    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = SiglipVisionModel.from_pretrained(MODEL).eval().to(device)

    # memmap avoids loading all features
    feats = np.lib.format.open_memmap(FEATS / "features.npy", mode="w+",
                                      dtype=np.float16, shape=(len(ids), 196, 768))
    with torch.no_grad():
        for start in range(0, len(ids), BATCH):
            chunk = ids[start:start + BATCH]
            imgs = [Image.open(COCO / files[i]).convert("RGB") for i in chunk]
            pixels = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            out = model(pixel_values=pixels).last_hidden_state
            assert out.shape[1:] == (196, 768), out.shape
            feats[start:start + len(chunk)] = out.float().cpu().numpy().astype(np.float16)
            if (start // BATCH) % 20 == 0:
                print(f"{start + len(chunk)}/{len(ids)}", flush=True)

    # save row index
    feats.flush()
    json.dump({str(i): row for row, i in enumerate(ids)}, open(FEATS / "index.json", "w"))
    # cost record for the report
    meta = {"images": len(ids), "seconds": round(time.time() - t0, 1), "device": device,
            "bytes": os.path.getsize(FEATS / "features.npy")}
    json.dump(meta, open(FEATS / "cache_meta.json", "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
