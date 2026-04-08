"""
Merge all extracted feature files into a single features.csv / features.json.

Source files (all under OUT_DIR):
  forget_set_features.csv   — per-triplet  (key: split)
  prompt_features.csv       — per-sample   (key: split + sample_index)
  interaction_features.csv  — per-sample   (key: split + test_index → renamed to sample_index)

The per-triplet features are broadcast to every sample within that triplet.
"""

import json
from pathlib import Path
import argparse

import pandas as pd

FEAT_DIR = Path(__file__).resolve().parent.parent

SAMPLE_FILES = [
    ("prompt_features.csv", None),
    ("interaction_features.csv", {"test_index": "sample_index"}),
]
TRIPLET_FILES = [
    "forget_set_features.csv",
]
SAMPLE_KEYS = ["split", "sample_index"]


def load_csv(feat_dir: Path, name: str, rename: dict | None = None) -> pd.DataFrame | None:
    path = feat_dir / name
    if not path.exists():
        print(f"  [SKIP] {name} not found")
        return None
    df = pd.read_csv(path)
    if rename:
        df.rename(columns=rename, inplace=True)
    print(f"  [OK]   {name:50s}  {df.shape[0]:>6} rows × {df.shape[1]} cols")
    return df


def merge(feat_dir: Path) -> pd.DataFrame:
    sample_dfs: list[pd.DataFrame] = []
    for name, rename in SAMPLE_FILES:
        df = load_csv(feat_dir, name, rename)
        if df is not None:
            sample_dfs.append(df)

    triplet_dfs: list[pd.DataFrame] = []
    for name in TRIPLET_FILES:
        df = load_csv(feat_dir, name)
        if df is not None:
            triplet_dfs.append(df)

    if not sample_dfs:
        raise RuntimeError("No per-sample feature files found — nothing to merge")

    merged = sample_dfs[0]
    for df in sample_dfs[1:]:
        dup_cols = [c for c in df.columns if c in merged.columns and c not in SAMPLE_KEYS]
        merged = merged.merge(df.drop(columns=dup_cols), on=SAMPLE_KEYS, how="outer")

    for df in triplet_dfs:
        dup_cols = [c for c in df.columns if c in merged.columns and c != "split"]
        merged = merged.merge(df.drop(columns=dup_cols), on="split", how="left")

    merged.sort_values(SAMPLE_KEYS, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge all feature files into features.csv / features.json")
    parser.add_argument("--feat_dir", type=str, default=str(FEAT_DIR))
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    print("=" * 72)
    print(f"  Merging features from: {feat_dir}")
    print("=" * 72)

    df = merge(feat_dir)

    csv_path = feat_dir / "features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  saved to {csv_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    json_path = feat_dir / "features.json"
    records = df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to {json_path}")

    print(f"\n── Columns ({df.shape[1]}) ──")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")

    print(f"\n── Preview (first 3 rows) ──")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
