"""
Train a Random Forest regressor to predict per-text ppl_ratio
(= unlearn_ppl / base_ppl) from dataset + text features.

Data:  training_data.csv  (5000 rows, 262 features, target: ppl_ratio)

Strategy:
  Leave-One-Group-Out CV (LOGO) where each group = eval_triplet, so we
  always evaluate on completely unseen test texts.  Grid search over
  RF hyperparameters, selecting the combo with the best R².

  Alternative grouping: --group_by model_triplet (unseen unlearn model).

Feature selection (--features):
  all               use every feature column
  col1,col2,...     explicit comma-separated column names
  cos_sim_*,eucl_*  fnmatch-style glob patterns (can mix with exact names)
"""

import argparse
import json
import warnings
from fnmatch import fnmatch
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
)
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
TRAINING_CSV = ROOT / "training_data.csv"

ID_COLS = {"model_triplet", "eval_triplet", "sample_index"}
LABEL_COLS = {"base_loss", "base_ppl", "unlearn_loss", "unlearn_ppl", "ppl_ratio"}
NON_FEATURE_COLS = ID_COLS | LABEL_COLS

TOP_N_IMPORTANCE = 30

PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", 0.5],
}


def resolve_feature_cols(spec: str, df: pd.DataFrame) -> list[str]:
    available = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if spec == "all":
        return available
    cols: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "*" in part or "?" in part:
            cols.extend(c for c in available if fnmatch(c, part))
        elif part in set(df.columns):
            cols.append(part)
        else:
            raise ValueError(f"Column not found: {part}")
    seen: set[str] = set()
    return [c for c in cols if c not in seen and not seen.add(c)]


def build_candidates():
    keys = list(PARAM_GRID.keys())
    candidates = []
    for vals in product(*PARAM_GRID.values()):
        params = dict(zip(keys, vals))
        params["random_state"] = 42
        params["n_jobs"] = -1
        label = ", ".join(f"{k}={v}" for k, v in params.items()
                          if k in PARAM_GRID)
        candidates.append((label, RandomForestRegressor(**params)))
    return candidates


def evaluate_logo(X, y, groups, model):
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(y, dtype=np.float64)
    for train_idx, test_idx in logo.split(X, y, groups):
        clone = RandomForestRegressor(**model.get_params())
        clone.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clone.predict(X[test_idx])
    return y_pred


def score_predictions(y, y_pred):
    residuals = y - y_pred
    return {
        "r2": r2_score(y, y_pred),
        "mse": mean_squared_error(y, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": mean_absolute_error(y, y_pred),
        "mape": float(np.mean(np.abs(residuals / np.clip(y, 1e-6, None)))) * 100,
        "median_ae": float(np.median(np.abs(residuals))),
        "max_ae": float(np.max(np.abs(residuals))),
    }


def print_section(title, width=70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RF regressor for per-text ppl_ratio prediction"
    )
    parser.add_argument("--data", default=str(TRAINING_CSV))
    parser.add_argument("--features", default="all",
                        help="all | col1,col2,... | glob patterns (default: all)")
    parser.add_argument("--target", default="ppl_ratio",
                        choices=["ppl_ratio", "unlearn_ppl", "unlearn_loss"],
                        help="Regression target (default: ppl_ratio)")
    parser.add_argument("--group_by", default="eval_triplet",
                        choices=["eval_triplet", "model_triplet"],
                        help="LOGO grouping column (default: eval_triplet)")
    parser.add_argument("--outdir", default=str(ROOT / "rf"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    feature_cols = resolve_feature_cols(args.features, df)
    df[feature_cols] = df[feature_cols].fillna(0)
    X = df[feature_cols].values
    y = df[args.target].values.astype(np.float64)
    groups = df[args.group_by].values

    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)

    print_section("Data Summary")
    print(f"  Feature set: {args.features}  ({len(feature_cols)} cols)")
    print(f"  Target:      {args.target}")
    print(f"  Samples:     {len(df)}")
    print(f"  Target stats: mean={y.mean():.4f}  std={y.std():.4f}  "
          f"min={y.min():.4f}  max={y.max():.4f}")
    print(f"  Group by:    {args.group_by}  ({len(set(groups))} groups)")
    print(f"  HP combos:   {n_combos}")

    # ── Grid search with LOGO CV ─────────────────────────────────────────
    print_section("LOGO CV Grid Search")
    candidates = build_candidates()
    best_r2 = -np.inf
    best_info = None

    for param_label, model in tqdm(candidates, desc="  Searching", ncols=90):
        y_pred = evaluate_logo(X, y, groups, model)
        scores = score_predictions(y, y_pred)
        if scores["r2"] > best_r2:
            best_r2 = scores["r2"]
            best_info = {
                "params": param_label,
                "scores": scores,
                "y_pred": y_pred,
                "model": model,
            }

    scores = best_info["scores"]
    y_pred = best_info["y_pred"]

    print(f"\n  Best params: {best_info['params']}")
    print(f"  LOGO CV metrics:")
    print(f"    R²:        {scores['r2']:.4f}")
    print(f"    RMSE:      {scores['rmse']:.4f}")
    print(f"    MAE:       {scores['mae']:.4f}")
    print(f"    MAPE:      {scores['mape']:.2f}%")
    print(f"    Median AE: {scores['median_ae']:.4f}")
    print(f"    Max AE:    {scores['max_ae']:.4f}")

    # ── Per-group breakdown ──────────────────────────────────────────────
    print_section(f"Per-Group Breakdown (group={args.group_by})")
    for group_name in sorted(set(groups)):
        mask = groups == group_name
        yt, yp = y[mask], y_pred[mask]
        r2 = r2_score(yt, yp) if len(yt) > 1 else 0.0
        mae = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        print(f"  {group_name:15s}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
              f"(n={mask.sum()}, mean_y={yt.mean():.4f})")

    # ── Self vs Cross breakdown ──────────────────────────────────────────
    print_section("Self vs Cross Breakdown")
    is_self = df["model_triplet"].values == df["eval_triplet"].values
    for label, mask in [("Self (model==eval)", is_self), ("Cross (model!=eval)", ~is_self)]:
        yt, yp = y[mask], y_pred[mask]
        r2 = r2_score(yt, yp) if len(yt) > 1 else 0.0
        mae = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        print(f"  {label:22s}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  (n={mask.sum()})")

    # ── Refit on all data ────────────────────────────────────────────────
    print_section("Refit on Full Data")
    model = best_info["model"]
    model.fit(X, y)

    importances = model.feature_importances_
    std_importances = np.std(
        [t.feature_importances_ for t in model.estimators_], axis=0
    )
    ranked = sorted(zip(feature_cols, importances, std_importances),
                    key=lambda x: x[1], reverse=True)

    print(f"  Top-{TOP_N_IMPORTANCE} feature importances (of {len(feature_cols)}):")
    for fname, imp, std in ranked[:TOP_N_IMPORTANCE]:
        bar = "#" * int(imp * 120)
        print(f"    {fname:>40s}  {imp:.4f} ±{std:.4f}  {bar}")

    # ── Save artifacts ───────────────────────────────────────────────────
    pred_df = df[["model_triplet", "eval_triplet", "sample_index", args.target]].copy()
    pred_df["pred"] = np.round(y_pred, 6)
    pred_df["residual"] = np.round(y - y_pred, 6)
    pred_df.to_csv(outdir / "logo_predictions.csv", index=False)

    model_path = outdir / "model_ppl.joblib"
    joblib.dump(model, model_path)

    results = {
        "task": "regression",
        "target": args.target,
        "cv": f"LeaveOneGroupOut (group={args.group_by})",
        "feature_set": args.features,
        "best_params": best_info["params"],
        "best_hyperparams": {
            k: v for k, v in model.get_params().items() if k in PARAM_GRID
        },
        "logo_cv_metrics": {k: round(v, 6) for k, v in scores.items()},
        "feature_importances": {
            fname: {"importance": round(float(imp), 6), "std": round(float(std), 6)}
            for fname, imp, std in ranked
        },
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }
    with open(outdir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_section("Summary")
    print(f"  LOGO CV  —  R²={scores['r2']:.4f}  "
          f"RMSE={scores['rmse']:.4f}  "
          f"MAE={scores['mae']:.4f}  "
          f"MAPE={scores['mape']:.2f}%")
    print(f"\n  Artifacts saved to {outdir}/")
    print(f"    - results.json          (metrics + all feature importances)")
    print(f"    - logo_predictions.csv  (per-text LOGO CV predictions)")
    print(f"    - model_ppl.joblib      (fitted model on all data)")


if __name__ == "__main__":
    main()
