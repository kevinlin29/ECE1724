#!/usr/bin/env python3
"""
Generate a JSONL file from MS MARCO v1.1 validation split.

Output format (one JSON per line):
{
  "query": "...",
  "passages": [...list of passage texts...],
  "labels": [...list of 0/1 relevance labels...]
}
"""

from datasets import load_dataset
from pathlib import Path
import json

OUTPUT_PATH = Path("msmarco_validation.jsonl")


def main():
    print("Loading MS MARCO v1.1 validation split...")
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

    print(f"Loaded {len(ds)} validation examples.")
    print(f"Writing to: {OUTPUT_PATH}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for sample in ds:
            query = sample["query"]
            passages = sample["passages"]["passage_text"]
            labels = sample["passages"]["is_selected"]  # list of 0/1 values

            obj = {
                "query": query,
                "passages": passages,
                "labels": labels
            }

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done!")
    print(f"Validation JSONL saved at: {OUTPUT_PATH.absolute()}")


if __name__ == "__main__":
    main()
