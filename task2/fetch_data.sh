#!/bin/bash
# download coco and vqav2 files
# extract then delete each zip
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)/data/task2"
RAW="$ROOT/raw"
mkdir -p "$RAW" "$ROOT/coco" "$ROOT/vqa"

fetch() {
  # args url and dest
  local url="$1" dest="$2"
  local name
  name=$(basename "$url")
  echo "[$(date +%H:%M:%S)] downloading $name"
  curl -fL -C - --retry 5 -o "$RAW/$name" "$url"
  echo "[$(date +%H:%M:%S)] extracting $name"
  unzip -q -o "$RAW/$name" -d "$dest"
  rm "$RAW/$name"
  echo "[$(date +%H:%M:%S)] done $name"
}

fetch http://images.cocodataset.org/annotations/annotations_trainval2014.zip "$ROOT/coco"
fetch http://images.cocodataset.org/zips/train2014.zip "$ROOT/coco"
fetch http://images.cocodataset.org/zips/val2014.zip "$ROOT/coco"

# vqa files from s3 mirror
for name in \
  v2_Questions_Train_mscoco \
  v2_Annotations_Train_mscoco \
  v2_Complementary_Pairs_Train_mscoco \
  v2_Questions_Val_mscoco \
  v2_Annotations_Val_mscoco \
  v2_Complementary_Pairs_Val_mscoco; do
  fetch "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/$name.zip" "$ROOT/vqa"
done

echo "[$(date +%H:%M:%S)] ALL DONE"
