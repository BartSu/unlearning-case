"""
Sample N triplets (train, val, test) from WikiText-103, adapted from
Algorithm 1 of "A Curious Case of Searching for the Correlation between
Training Data and Adversarial Robustness of Transformer Textual Models"
(Dang et al., ACL 2024).

Key adaptation: each triplet has a *different* forget-set (train) size,
linearly spaced from --train_min to --train_max, to create structural
variation in both dataset features (X) and unlearning damage (Y).

  1. Filter: remove heading lines (= ... =) and short fragments (< min_chars).
  2. Sample a fixed S_test of size --test_size (shared across triplets).
  3. For i in [1..N]:
       Compute train_size_i = train_min + (i-1) * step.
       Randomly sample S_train (size train_size_i) and S_val (size --val_size)
       from the pool, with S_train ∩ S_val = ∅ and both ∩ S_test = ∅.

Output structure:
  wikitext/
    test.json                  (fixed test set, shared across triplets)
    triplet_001/train.json     (smallest forget set)
    triplet_001/val.json
    ...
    triplet_100/train.json     (largest forget set)
    triplet_100/val.json
    manifest.json              (triplet metadata: sizes, seed, etc.)
"""

import json
import os
import re
import random
import argparse

from datasets import load_dataset

HEADING_RE = re.compile(r"^\s*=+\s.*\s=+\s*$")


def load_wikitext_passages(min_chars=50):
    """Load WikiText-103, keeping only non-heading paragraphs >= min_chars."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    passages = []
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        if HEADING_RE.match(text):
            continue
        if len(text) >= min_chars:
            passages.append(text)
    return passages


def compute_train_sizes(n_triplets, train_min, train_max):
    """Linearly spaced forget-set sizes from train_min to train_max."""
    if n_triplets == 1:
        return [train_min]
    step = (train_max - train_min) / (n_triplets - 1)
    return [round(train_min + i * step) for i in range(n_triplets)]


def main():
    parser = argparse.ArgumentParser(
        description="Sample N variable-size triplets from WikiText-103"
    )
    parser.add_argument("--N", type=int, default=100,
                        help="Number of triplets to sample")
    parser.add_argument("--test_size", type=int, default=200,
                        help="Fixed test set size (shared)")
    parser.add_argument("--val_size", type=int, default=200,
                        help="Fixed val (retain) set size per triplet")
    parser.add_argument("--train_min", type=int, default=100,
                        help="Smallest forget set size (triplet_001)")
    parser.add_argument("--train_max", type=int, default=5050,
                        help="Largest forget set size (triplet_N)")
    parser.add_argument("--min_chars", type=int, default=50,
                        help="Min character length to keep a paragraph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    train_sizes = compute_train_sizes(args.N, args.train_min, args.train_max)

    print(f"Config: N={args.N}, test={args.test_size}, val={args.val_size}, "
          f"train=[{args.train_min}..{args.train_max}], seed={args.seed}")
    print("Loading WikiText-103 (filtering headings + short fragments) ...")
    passages = load_wikitext_passages(min_chars=args.min_chars)
    total = len(passages)
    print(f"Loaded {total:,} passages (min_chars={args.min_chars})")

    max_needed = max(train_sizes) + args.val_size
    pool_size = total - args.test_size
    assert pool_size >= max_needed, (
        f"Pool ({pool_size:,}) < max train+val ({max_needed:,}). "
        f"Reduce train_max or min_chars."
    )

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "wikitext"
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: fixed test set ---
    all_indices = list(range(total))
    random.shuffle(all_indices)
    test_indices = sorted(all_indices[: args.test_size])
    remaining_pool = all_indices[args.test_size :]

    test_data = [{"text": passages[i]} for i in test_indices]
    test_path = os.path.join(output_dir, "test.json")
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Fixed test set: {len(test_data)} passages -> {test_path}")

    # --- Step 2: N triplets with variable train size ---
    manifest = {
        "seed": args.seed,
        "total_passages": total,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "n_triplets": args.N,
        "triplets": [],
    }

    for n in range(1, args.N + 1):
        ts = train_sizes[n - 1]
        random.shuffle(remaining_pool)
        train_indices = sorted(remaining_pool[:ts])
        val_indices = sorted(remaining_pool[ts : ts + args.val_size])

        train_data = [{"text": passages[i]} for i in train_indices]
        val_data = [{"text": passages[i]} for i in val_indices]

        triplet_dir = os.path.join(output_dir, f"triplet_{n:03d}")
        os.makedirs(triplet_dir, exist_ok=True)

        with open(os.path.join(triplet_dir, "train.json"), "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        with open(os.path.join(triplet_dir, "val.json"), "w") as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        manifest["triplets"].append({
            "name": f"triplet_{n:03d}",
            "train_size": len(train_data),
            "val_size": len(val_data),
        })

        print(f"  triplet_{n:03d}: train={len(train_data):>5d}, val={len(val_data)}")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {args.N} triplets + 1 shared test set saved to {output_dir}/")
    print(f"  Train sizes: {train_sizes[0]} -> {train_sizes[-1]} "
          f"(step ≈ {train_sizes[1] - train_sizes[0] if args.N > 1 else 0})")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
