"""
Build per-text training data for the PPL predictor.

Joins two sources:
  1. Features  — feature-engineering/features.csv
     (per test-text, keyed by split + sample_index)
  2. PPL labels — extract-ppl/wikitext_cross_metrics_detail.json
     (per-sample base & unlearn loss/ppl from test split)

Each row = one (model_triplet, eval_triplet, sample_index) tuple.
  100 cross-eval pairs × 50 test samples = 5000 rows.

Columns: model_triplet, eval_triplet, sample_index,
         [all features], base_loss, base_ppl, unlearn_loss, unlearn_ppl,
         ppl_ratio (= unlearn_ppl / base_ppl)

Output: ppl-predictor/training_data.csv  +  training_data.json (metadata)
"""

import json
from pathlib import Path
import argparse

import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES_CSV = ROOT.parent / "feature-engineering" / "features.csv"
DETAIL_JSON = ROOT.parent / "extract-ppl" / "wikitext_cross_metrics_detail.json"

ID_COLS = ["model_triplet", "eval_triplet", "sample_index"]
LABEL_COLS = ["base_loss", "base_ppl", "unlearn_loss", "unlearn_ppl", "ppl_ratio"]
NON_FEATURE_COLS = set(ID_COLS + LABEL_COLS)


def load_ppl_labels(detail_path: Path) -> pd.DataFrame:
    with open(detail_path) as f:
        data = json.load(f)

    rows = []
    for result in data["results"]:
        model_t = result["model_triplet"]
        eval_t = result["eval_triplet"]
        base_samples = result["base"].get("test", [])
        unlearn_samples = result["unlearn"].get("test", [])

        if not base_samples or not unlearn_samples:
            continue
        if len(base_samples) != len(unlearn_samples):
            print(f"  WARNING: {model_t} x {eval_t}: base({len(base_samples)}) != unlearn({len(unlearn_samples)}), skipping")
            continue

        for idx, (b, u) in enumerate(zip(base_samples, unlearn_samples)):
            b_ppl = b["ppl"]
            u_ppl = u["ppl"]
            rows.append({
                "model_triplet": model_t,
                "eval_triplet": eval_t,
                "sample_index": idx,
                "base_loss": b["loss"],
                "base_ppl": b_ppl,
                "unlearn_loss": u["loss"],
                "unlearn_ppl": u_ppl,
                "ppl_ratio": round(u_ppl / max(b_ppl, 1e-6), 6),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build training data: features + per-sample PPL -> training_data.csv"
    )
    parser.add_argument("--features", default=str(FEATURES_CSV))
    parser.add_argument("--detail", default=str(DETAIL_JSON))
    parser.add_argument("--output", default=str(ROOT / "training_data.csv"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load features ─────────────────────────────────────────────────
    feat_df = pd.read_csv(args.features)
    feature_cols = [c for c in feat_df.columns if c not in NON_FEATURE_COLS and c not in ("split", "sample_index")]
    print(f"Features: {len(feat_df)} rows, {len(feature_cols)} feature cols")

    # ── 2. Load PPL labels ───────────────────────────────────────────────
    label_df = load_ppl_labels(Path(args.detail))
    n_pairs = label_df.groupby(["model_triplet", "eval_triplet"]).ngroups
    print(f"PPL labels: {len(label_df)} rows from {n_pairs} cross-eval pairs")

    # ── 3. Join features (by eval_triplet + sample_index) ────────────────
    feat_df = feat_df.rename(columns={"split": "eval_triplet"})
    join_keys = ["eval_triplet", "sample_index"]

    df = label_df.merge(feat_df, on=join_keys, how="inner")

    n_labels_only = len(label_df) - len(df)
    if n_labels_only:
        print(f"  {n_labels_only} label rows had no matching features (skipped)")

    # ── 4. Output ────────────────────────────────────────────────────────
    out_cols = ID_COLS + feature_cols + LABEL_COLS
    out_cols = [c for c in out_cols if c in df.columns]
    df = df[out_cols].sort_values(ID_COLS).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Training data saved to {output_path}")
    print(f"  Total samples:    {len(df)}")
    print(f"  Features:         {len(feature_cols)}")
    print(f"  Model triplets:   {df['model_triplet'].nunique()}")
    print(f"  Eval triplets:    {df['eval_triplet'].nunique()}")
    print(f"  Cross-eval pairs: {df.groupby(['model_triplet', 'eval_triplet']).ngroups}")
    print(f"  base_ppl  — mean={df['base_ppl'].mean():.2f}  std={df['base_ppl'].std():.2f}")
    print(f"  unlearn_ppl — mean={df['unlearn_ppl'].mean():.2f}  std={df['unlearn_ppl'].std():.2f}")
    print(f"  ppl_ratio — mean={df['ppl_ratio'].mean():.4f}  std={df['ppl_ratio'].std():.4f}")
    print(f"{'=' * 60}")

    json_path = output_path.with_suffix(".json")
    meta = {
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "id_columns": ID_COLS,
        "label_columns": LABEL_COLS,
        "model_triplets": sorted(df["model_triplet"].unique().tolist()),
        "eval_triplets": sorted(df["eval_triplet"].unique().tolist()),
        "stats": {
            "base_ppl_mean": round(df["base_ppl"].mean(), 4),
            "unlearn_ppl_mean": round(df["unlearn_ppl"].mean(), 4),
            "ppl_ratio_mean": round(df["ppl_ratio"].mean(), 4),
            "ppl_ratio_std": round(df["ppl_ratio"].std(), 4),
        },
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved to {json_path}")


if __name__ == "__main__":
    main()
