"""
Train Random Forest regressors to predict unlearning corruption from
dataset features.

Data:
  - Features (X): ../feature-engineering/wikitext_features.csv  (100 triplets × 13 features)
  - Labels   (Y): ../extract-label/wikitext_labels.csv

Targets:
  Y_general   – general test-set loss delta
  Y_forget    – forget-set loss delta
  Y_retain    – retain-set loss delta
  Y_precision – precision metric

Strategy:
  LOOCV over RandomForestRegressor with conservative hyperparameters
  (limited depth, higher min_samples_leaf) to avoid overfitting on ~28
  labeled samples. Best model per target is refit on all labeled data,
  then used to predict all 100 triplets.
"""

import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
FEATURES_CSV = ROOT.parent / "feature-engineering" / "wikitext_features.csv"
LABELS_CSV = ROOT.parent / "extract-label" / "wikitext_labels.csv"

FEATURE_COLS = [
    "mean_distance_among_classes (MD)",
    "fishers_discriminant_ratio (FDR)",
    "calinski_harabasz_index (CHI)",
    "davies_bouldin_index (DBI)",
    "n_clusters (HDBSCAN)",
    "pearson_median_skewness (PMS)",
    "kurtosis (Kurt)",
    "n_classes",
    "misclassification_rate (MCR)",
    "avg_n_tokens",
    "min_n_tokens",
    "max_n_tokens",
    "n_unique_tokens",
]

TARGET_COLS = ["Y_general", "Y_forget", "Y_retain", "Y_precision"]

PARAM_GRID = {
    "n_estimators": [200],
    "max_depth": [2, 3, 5],
    "min_samples_leaf": [2, 3, 5],
    "max_features": ["sqrt", 0.5, 1.0],
}


def load_data(features_path: str, labels_path: str):
    feat = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    merged = feat.merge(labels, on="split", how="inner")
    return feat, labels, merged


def build_candidates():
    """Generate all RF hyperparameter combinations."""
    keys = list(PARAM_GRID.keys())
    candidates = []
    for vals in product(*PARAM_GRID.values()):
        params = dict(zip(keys, vals))
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        candidates.append((label, model))
    return candidates


def evaluate_loocv(X: np.ndarray, y: np.ndarray, model):
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse, y_pred


def select_best(X, y, candidates, target_name=""):
    best_r2 = -np.inf
    best_info = None

    for param_label, model in tqdm(candidates, desc=f"  LOOCV search [{target_name}]",
                                   leave=True, ncols=90):
        r2, mae, rmse, y_pred = evaluate_loocv(X, y, model)
        if r2 > best_r2:
            best_r2 = r2
            best_info = {
                "params": param_label,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "y_pred": y_pred,
                "model": model,
            }

    return best_info


def print_section(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main():
    parser = argparse.ArgumentParser(description="Train RF corruption predictors")
    parser.add_argument("--features", default=str(FEATURES_CSV))
    parser.add_argument("--labels", default=str(LABELS_CSV))
    parser.add_argument("--outdir", default=str(ROOT / "rf"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feat_all, labels, merged = load_data(args.features, args.labels)

    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)

    print_section("Data Summary")
    print(f"  Feature samples (all):    {len(feat_all)}")
    print(f"  Labeled samples:          {len(labels)}")
    print(f"  Merged (train) samples:   {len(merged)}")
    print(f"  Features:                 {len(FEATURE_COLS)}")
    print(f"  Targets:                  {TARGET_COLS}")
    print(f"  Hyperparam combinations:  {n_combos}")

    X_train = merged[FEATURE_COLS].values
    candidates = build_candidates()

    results = {}
    models = {}
    loocv_predictions = merged[["split"]].copy()

    for target in TARGET_COLS:
        y = merged[target].values
        print_section(f"Target: {target}")
        print(f"  Range: [{y.min():.6f}, {y.max():.6f}]  Mean: {y.mean():.6f}  Std: {y.std():.6f}")

        best = select_best(X_train, y, candidates, target_name=target)
        print(f"\n  Best model:  RF ({best['params']})")
        print(f"  LOOCV R²:    {best['r2']:.4f}")
        print(f"  LOOCV MAE:   {best['mae']:.6f}")
        print(f"  LOOCV RMSE:  {best['rmse']:.6f}")

        model = best["model"]
        model.fit(X_train, y)
        models[target] = model

        loocv_predictions[f"{target}_actual"] = y
        loocv_predictions[f"{target}_pred_loocv"] = best["y_pred"]
        loocv_predictions[f"{target}_error"] = y - best["y_pred"]

        importances = model.feature_importances_
        print(f"\n  Feature importances:")
        ranked = sorted(
            zip(FEATURE_COLS, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        for fname, imp in ranked:
            bar = "#" * int(imp * 60)
            print(f"    {fname:>42s}  {imp:.4f}  {bar}")

        # per-tree variability of feature importances
        std_importances = np.std(
            [t.feature_importances_ for t in model.estimators_], axis=0
        )
        results[target] = {
            "model": "RandomForest",
            "params": best["params"],
            "loocv_r2": round(best["r2"], 6),
            "loocv_mae": round(best["mae"], 6),
            "loocv_rmse": round(best["rmse"], 6),
            "best_hyperparams": {
                k: v for k, v in model.get_params().items()
                if k in PARAM_GRID
            },
            "feature_importances": {
                fname: round(float(imp), 6)
                for fname, imp in zip(FEATURE_COLS, importances)
            },
            "feature_importances_std": {
                fname: round(float(s), 6)
                for fname, s in zip(FEATURE_COLS, std_importances)
            },
        }

    loocv_path = outdir / "loocv_predictions.csv"
    loocv_predictions.to_csv(loocv_path, index=False)
    print_section("LOOCV Predictions")
    print(f"  Saved to {loocv_path}")

    X_all = feat_all[FEATURE_COLS].values
    predictions = feat_all[["split"]].copy()
    for target in TARGET_COLS:
        predictions[target] = models[target].predict(X_all)

    pred_path = outdir / "wikitext_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    print_section("Full Predictions (100 triplets)")
    print(f"  Saved to {pred_path}")
    print(f"\n{predictions.describe().to_string()}")

    for target in TARGET_COLS:
        model_path = outdir / f"model_{target}.joblib"
        joblib.dump(models[target], model_path)

    results_path = outdir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print_section("Summary")
    print(f"  {'Target':<16s}  {'R²':>8s}  {'MAE':>10s}  {'RMSE':>10s}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*10}  {'-'*10}")
    for target in TARGET_COLS:
        r = results[target]
        print(f"  {target:<16s}  {r['loocv_r2']:>8.4f}  {r['loocv_mae']:>10.6f}  {r['loocv_rmse']:>10.6f}")

    print(f"\n  Artifacts saved to: {outdir}")
    print(f"    - results.json              (metrics + feature importances)")
    print(f"    - loocv_predictions.csv     (leave-one-out predictions)")
    print(f"    - wikitext_predictions.csv  (all 100 triplet predictions)")
    print(f"    - model_<target>.joblib     (fitted models)")


if __name__ == "__main__":
    main()
