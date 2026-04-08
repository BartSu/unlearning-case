"""
Build per-question training data for the corruption predictor.

Joins two sources:
  1. Features  — feature-engineering/features.csv  (per-question, keyed by
                 split + source_train_index)
  2. Labels    — extract-label/wikitext_qa_labels/pairs/*.json  (per-question,
                 base_correct / unlearn_correct / label / case)

Only base-correct samples are kept (WW / WC excluded).
  label = 1  CW  (base correct, unlearn wrong)
  label = 0  CC  (base correct, unlearn correct)

Output: corruption-predictor/training_data.csv
"""

import json
from pathlib import Path
import argparse

import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES_CSV = ROOT.parent / "feature-engineering" / "features.csv"
PAIRS_DIR = ROOT.parent / "extract-label" / "wikitext_qa_labels" / "pairs"

ID_COLS = ["split", "source_train_index", "question"]
LABEL_COLS = ["base_correct", "unlearn_correct", "case", "label"]
NON_FEATURE_COLS = set(ID_COLS + LABEL_COLS)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_labels(pairs_dir: Path) -> pd.DataFrame:
    pair_files = sorted(pairs_dir.glob("*.json"))
    if not pair_files:
        raise FileNotFoundError(f"No pair files in {pairs_dir}")

    rows = []
    n_excluded = 0
    for pf in pair_files:
        pair = load_json(pf)
        triplet = pair["eval_triplet"]
        for ex in pair["examples"]:
            if not ex["base_correct"]:
                n_excluded += 1
                continue
            rows.append({
                "split": triplet,
                "source_train_index": ex["source_train_index"],
                "base_correct": ex["base_correct"],
                "unlearn_correct": ex["unlearn_correct"],
                "case": ex["case"],
                "label": 1 if ex["case"] == "base_correct_unlearn_wrong" else 0,
            })
    if n_excluded:
        print(f"  Excluded {n_excluded} base-wrong samples (WW + WC)")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build training data: features (CSV) + labels (pair JSON) -> training_data.csv"
    )
    parser.add_argument("--features", default=str(FEATURES_CSV))
    parser.add_argument("--pairs_dir", default=str(PAIRS_DIR))
    parser.add_argument("--output", default=str(ROOT / "training_data.csv"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load features ─────────────────────────────────────────────────────
    feat_df = pd.read_csv(args.features)
    feature_cols = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]
    print(f"Features: {len(feat_df)} rows, {len(feature_cols)} feature cols")

    # ── 2. Load labels ───────────────────────────────────────────────────────
    label_df = load_labels(Path(args.pairs_dir))
    print(f"Labels:   {len(label_df)} rows from {label_df['split'].nunique()} triplets")

    # ── 3. Join ──────────────────────────────────────────────────────────────
    join_keys = ["split", "source_train_index"]
    df = feat_df.merge(label_df, on=join_keys, how="inner")

    n_features_only = len(feat_df) - len(df)
    n_labels_only = len(label_df) - len(df)
    if n_features_only:
        print(f"  {n_features_only} feature rows had no matching label (skipped)")
    if n_labels_only:
        print(f"  {n_labels_only} label rows had no matching feature (skipped)")

    out_cols = ID_COLS + feature_cols + LABEL_COLS
    out_cols = [c for c in out_cols if c in df.columns]
    df = df[out_cols].sort_values(join_keys).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    print(f"\n{'=' * 60}")
    print(f"  Training data saved to {output_path}")
    print(f"  Total samples:  {len(df)}")
    print(f"  Label=1 (CW):   {n_pos}  ({n_pos/len(df):.1%})")
    print(f"  Label=0 (CC):   {n_neg}  ({n_neg/len(df):.1%})")
    print(f"  Features:       {len(feature_cols)}")
    print(f"  Triplets:       {df['split'].nunique()}")
    print(f"{'=' * 60}")

    json_path = output_path.with_suffix(".json")
    meta = {
        "n_samples": len(df),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": round(n_pos / len(df), 4) if len(df) else 0,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "triplets": sorted(df["split"].unique().tolist()),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  Metadata saved to {json_path}")


if __name__ == "__main__":
    main()
