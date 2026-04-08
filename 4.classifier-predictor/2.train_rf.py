"""
Train a Random Forest classifier to predict per-question corruption
(CW: base correct -> unlearn wrong) from dataset + question features.

Data:  training_data.csv  (per-question rows, label ∈ {0, 1})

Strategy:
  Leave-One-Group-Out CV (LOGO) where each group = triplet (split), so we
  always evaluate on a completely unseen forgetting set.  Grid search over
  RF hyperparameters, selecting the combo with the best macro-F1.

Feature selection (--features):
  all               use every feature column in training_data.csv (~375)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
TRAINING_CSV = ROOT / "training_data.csv"

NON_FEATURE_COLS = {
    "split", "source_train_index", "question",
    "base_correct", "unlearn_correct", "case", "label",
}
TOP_N_IMPORTANCE = 30

PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, 8, None],
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
        params["class_weight"] = "balanced"
        params["random_state"] = 42
        params["n_jobs"] = -1
        label = ", ".join(f"{k}={v}" for k, v in params.items()
                          if k in PARAM_GRID)
        candidates.append((label, RandomForestClassifier(**params)))
    return candidates


def evaluate_logo(X, y, groups, model):
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(y)
    y_prob = np.empty(len(y), dtype=np.float64)
    for train_idx, test_idx in logo.split(X, y, groups):
        clone = RandomForestClassifier(**model.get_params())
        clone.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clone.predict(X[test_idx])
        y_prob[test_idx] = clone.predict_proba(X[test_idx])[:, 1]
    return y_pred, y_prob


def score_predictions(y, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob) if len(set(y)) > 1 else 0.0,
    }


def print_section(title, width=70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RF classifier for per-question corruption prediction"
    )
    parser.add_argument("--data", default=str(TRAINING_CSV))
    parser.add_argument("--features", default="all",
                        help="all | col1,col2,... | glob patterns (default: all)")
    parser.add_argument("--outdir", default=str(ROOT / "rf"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    feature_cols = resolve_feature_cols(args.features, df)
    df[feature_cols] = df[feature_cols].fillna(0)
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["split"].values

    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)

    print_section("Data Summary")
    print(f"  Feature set: {args.features}  ({len(feature_cols)} cols)")
    print(f"  Samples:     {len(df)}")
    print(f"  Label=1:     {y.sum()}  ({y.mean():.1%})")
    print(f"  Label=0:     {(1-y).sum():.0f}  ({1-y.mean():.1%})")
    print(f"  Groups:      {len(set(groups))} triplets")
    print(f"  HP combos:   {n_combos}")

    # ── Grid search with LOGO CV ─────────────────────────────────────────
    print_section("LOGO CV Grid Search")
    candidates = build_candidates()
    best_f1 = -1.0
    best_info = None

    for param_label, model in tqdm(candidates, desc="  Searching", ncols=90):
        y_pred, y_prob = evaluate_logo(X, y, groups, model)
        scores = score_predictions(y, y_pred, y_prob)
        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            best_info = {
                "params": param_label,
                "scores": scores,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "model": model,
            }

    scores = best_info["scores"]
    y_pred = best_info["y_pred"]
    y_prob = best_info["y_prob"]

    print(f"\n  Best params: {best_info['params']}")
    print(f"  LOGO CV metrics:")
    print(f"    Accuracy:  {scores['accuracy']:.4f}")
    print(f"    Precision: {scores['precision']:.4f}")
    print(f"    Recall:    {scores['recall']:.4f}")
    print(f"    F1:        {scores['f1']:.4f}")
    print(f"    ROC-AUC:   {scores['roc_auc']:.4f}")

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred)
    print(f"\n  Confusion matrix (LOGO CV):")
    print(f"                 Predicted 0  Predicted 1")
    print(f"    Actual 0     {cm[0,0]:>8d}     {cm[0,1]:>8d}")
    print(f"    Actual 1     {cm[1,0]:>8d}     {cm[1,1]:>8d}")

    print(f"\n  Classification report:")
    print(classification_report(y, y_pred, target_names=["Normal (0)", "Forgotten (1)"], digits=4))

    # ── Per-group breakdown ──────────────────────────────────────────────
    print_section("Per-Triplet Breakdown (LOGO CV)")
    for triplet in sorted(set(groups)):
        mask = groups == triplet
        yt, yp = y[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        f1 = f1_score(yt, yp, zero_division=0)
        print(f"  {triplet:15s}  TP={tp:2d} FN={fn:2d} FP={fp:2d} TN={tn:2d}  "
              f"F1={f1:.4f}  (n={mask.sum()}, pos={yt.sum()})")

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
        bar = "#" * int(imp * 60)
        print(f"    {fname:>40s}  {imp:.4f} ±{std:.4f}  {bar}")

    # ── Save artifacts ───────────────────────────────────────────────────
    pred_df = df[["split", "source_train_index", "label"]].copy()
    pred_df["pred_label"] = y_pred
    pred_df["pred_prob"] = np.round(y_prob, 6)
    pred_df.to_csv(outdir / "logo_predictions.csv", index=False)

    model_path = outdir / "model_corruption.joblib"
    joblib.dump(model, model_path)

    results = {
        "task": "binary_classification",
        "target": "label (CW=1, else=0)",
        "cv": "LeaveOneGroupOut (group=triplet)",
        "class_weight": "balanced",
        "feature_set": args.features,
        "best_params": best_info["params"],
        "best_hyperparams": {
            k: v for k, v in model.get_params().items() if k in PARAM_GRID
        },
        "logo_cv_metrics": {k: round(v, 6) for k, v in scores.items()},
        "confusion_matrix": {"TP": int(cm[1, 1]), "FN": int(cm[1, 0]),
                             "FP": int(cm[0, 1]), "TN": int(cm[0, 0])},
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
    print(f"  LOGO CV  —  F1={scores['f1']:.4f}  "
          f"Prec={scores['precision']:.4f}  "
          f"Rec={scores['recall']:.4f}  "
          f"AUC={scores['roc_auc']:.4f}")
    print(f"\n  Artifacts saved to {outdir}/")
    print(f"    - results.json            (metrics + all feature importances)")
    print(f"    - logo_predictions.csv    (per-question LOGO CV predictions)")
    print(f"    - model_corruption.joblib (fitted model on all data)")


if __name__ == "__main__":
    main()
