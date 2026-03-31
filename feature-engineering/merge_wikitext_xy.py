"""
Merge WikiText features (X) and unlearning labels (Y) into a single dataset
for training the robustness/unlearning predictor.

Input:
  - wikitext_features.csv:  13 features per triplet  (X)
  - wikitext_labels.csv:    loss delta per triplet    (Y)
Output:
  - wikitext_dataset.csv:   merged X + Y
"""

import argparse
from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(OUT_DIR / "wikitext_features.csv"))
    parser.add_argument("--labels", default=str(OUT_DIR / "wikitext_labels.csv"))
    parser.add_argument("--output", default=str(OUT_DIR / "wikitext_dataset.csv"))
    args = parser.parse_args()

    feat = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)

    merged = feat.merge(labels, on="split", how="inner")
    print(f"Features: {len(feat)} rows")
    print(f"Labels:   {len(labels)} rows")
    print(f"Merged:   {len(merged)} rows")

    merged.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(f"\nColumns: {list(merged.columns)}")
    print(f"\n{merged.describe().to_string()}")


if __name__ == "__main__":
    main()
