#!/usr/bin/env python3
"""
Fetch and persist the 20 Newsgroups dataset using scikit-learn.

DATA SOURCE:
    sklearn.datasets.fetch_20newsgroups
    Derived from the original CMU 20 Newsgroups collection:
    Lang, K. (1995). Newsweeder: Learning to filter netnews.
    Proceedings of the 12th International Conference on Machine Learning (ICML 1995).

    - ~18,000 posts across 20 Usenet newsgroups (1990s)
    - Train: 11,314 | Test: 7,532
    - Categories: comp.*, rec.*, sci.*, talk.*, misc.*, alt.*, soc.*

OUTPUT:
    data/raw/20newsgroups_train.json    — Training documents with metadata
    data/raw/20newsgroups_test.json     — Test documents with metadata
    data/raw/20newsgroups_stats.csv     — Dataset statistics (class distribution, lengths)
"""

import json
import csv
import os
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups


def fetch_and_save(subset: str, out_dir: Path) -> dict:
    """Fetch a subset, save to JSON, return stats."""
    print(f"  Fetching 20newsgroups subset='{subset}'...")
    data = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    records = []
    for i, (text, target) in enumerate(zip(data.data, data.target)):
        records.append(
            {
                "id": i,
                "text": text,
                "target": int(target),
                "target_name": data.target_names[target],
            }
        )

    out_path = out_dir / f"20newsgroups_{subset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "subset": subset,
                "n_samples": len(records),
                "target_names": data.target_names,
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"    ✓ Saved {len(records):,} records → {out_path}")

    # Per-class counts
    class_counts = {name: 0 for name in data.target_names}
    text_lengths = []
    for r in records:
        class_counts[r["target_name"]] += 1
        text_lengths.append(len(r["text"].split()))

    return {
        "subset": subset,
        "n_samples": len(records),
        "n_classes": len(data.target_names),
        "avg_length": round(sum(text_lengths) / len(text_lengths), 1) if text_lengths else 0,
        "min_length": min(text_lengths) if text_lengths else 0,
        "max_length": max(text_lengths) if text_lengths else 0,
        "class_counts": class_counts,
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Fetching REAL 20 Newsgroups Dataset")
    print("Source: sklearn.datasets.fetch_20newsgroups")
    print("=" * 60)

    train_stats = fetch_and_save("train", raw_dir)
    test_stats = fetch_and_save("test", raw_dir)

    # Write statistics CSV
    stats_path = raw_dir / "20newsgroups_stats.csv"
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "n_samples", "n_classes", "avg_length", "min_length", "max_length"])
        for s in (train_stats, test_stats):
            writer.writerow(
                [s["subset"], s["n_samples"], s["n_classes"], s["avg_length"], s["min_length"], s["max_length"]]
            )
        writer.writerow([])
        writer.writerow(["subset", "class_name", "count"])
        for s in (train_stats, test_stats):
            for cls, cnt in s["class_counts"].items():
                writer.writerow([s["subset"], cls, cnt])

    print(f"    ✓ Saved statistics → {stats_path}")
    print()
    print(f"Train: {train_stats['n_samples']:,} docs | Test: {test_stats['n_samples']:,} docs")
    print(f"Classes: {train_stats['n_classes']} | Avg length: {train_stats['avg_length']} tokens")
    print()
    print("=" * 60)
    print("Fetch complete — all data is REAL (sklearn 20 Newsgroups)")
    print("=" * 60)


if __name__ == "__main__":
    main()
