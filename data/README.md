# data

`task2/manifests/` is committed. It holds the split files, example IDs and pools used by Task 2. Each JSONL row has the image ID, question and answers.

Not committed (download or build locally):

| Path | Size | How to get it |
|---|---|---|
| `task2/coco/` | about 20 GB | `bash task2/fetch_data.sh` |
| `task2/vqa/` | about 0.6 GB | `bash task2/fetch_data.sh` |
| `task2/features/` | about 5.6 GB | `.venv/bin/python task2/cache_features.py mps` |

Task 1 data (MetaMathQA, GSM8K) is downloaded from Hugging Face by the Kaggle notebooks. Local copies go to `cache/`.
