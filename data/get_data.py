from datasets import load_dataset
import json

# Load triplets with actual text - "triplet" subset has strings
ds = load_dataset("sentence-transformers/msmarco-msmarco-MiniLM-L6-v3", "triplet", split="train[:10000]")

# Convert to your required format
with open("train.jsonl", "w") as f:
    for row in ds:
        obj = {
            "query": row["query"],
            "positives": [row["positive"]],
            "negatives": [row["negative"]]
        }
        f.write(json.dumps(obj) + "\n")