"""
Predict per-sample unlearning corruption (log ppl_ratio) from
*pure embedding geometry* between a forget set and a target text.

Motivation
----------
The 261-feature regressor is dominated by per-text surface statistics
(char entropy, punctuation count) that have nothing to do with the forget
set — it is essentially a "text-hardness" predictor, not a data-first
predictor of corruption.

If the three-layer corruption story (L1 forget → L2 locality → L3
spillover) is a geometric phenomenon, then embedding-relationship features
between (forget_set, target_text) should suffice to explain a useful
fraction of the variance in log(ppl_ratio) — and without a per-text
surface feature in sight.

Features (16 geometric, no lexical/surface statistics)
------------------------------------------------------
Target ↔ forget:
  cos_sim_to_centroid         cos_sim_to_nearest
  cos_sim_top3_mean           cos_sim_top5_mean
  cos_sim_mean                cos_sim_std
  eucl_to_centroid            eucl_to_nearest
  proj_on_centroid            angle_to_centroid
Forget-set intrinsic geometry:
  forget_emb_variance_mean    forget_mean_pairwise_similarity
  forget_centroid_norm        forget_spread (mean pairwise euclidean)
Target intrinsic:
  target_emb_norm
Binary:
  same_cluster                (forget_triplet == eval_triplet)

Target: log(unlearn_ppl / base_ppl) on the test split.

Evaluation: Leave-One-Group-Out CV by eval_triplet.
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TRIPLET_DIR = ROOT / "1.data-preparation" / "data" / "wikitext_hdbscan_triplets"
CROSS_JSON = ROOT / "2.extract-ppl" / "wikitext_cross_metrics_detail.json"
OUT_DIR = Path(__file__).resolve().parent / "geometry"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def load_texts(triplet: str, split: str) -> list[str]:
    with open(TRIPLET_DIR / triplet / f"{split}.json") as f:
        return [x["text"] for x in json.load(f)]


def build_features() -> pd.DataFrame:
    with open(CROSS_JSON) as f:
        cross = json.load(f)

    triplets = sorted({r["model_triplet"] for r in cross["results"]} |
                      {r["eval_triplet"]  for r in cross["results"]})
    print(f"Clusters (triplets): {len(triplets)} → {triplets}")

    print("Loading sentence-transformer (all-MiniLM-L6-v2) ...")
    enc = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode all forget sets and all test texts once.
    forget_embs: dict[str, np.ndarray] = {}
    forget_geom: dict[str, dict] = {}
    for t in triplets:
        txts = load_texts(t, "train")
        embs = enc.encode(txts, show_progress_bar=False, batch_size=64)
        forget_embs[t] = embs
        centroid = embs.mean(axis=0)
        # intrinsic geometry
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        sim = cosine_similarity(embs)
        dist = euclidean_distances(embs)
        triu = np.triu_indices(len(embs), k=1)
        forget_geom[t] = {
            "centroid": centroid,
            "centroid_norm": float(np.linalg.norm(centroid)),
            "emb_variance_mean": float(embs.var(axis=0).mean()),
            "mean_pairwise_similarity": float(sim[triu].mean()),
            "spread": float(dist[triu].mean()),
        }

    test_embs: dict[str, np.ndarray] = {}
    test_texts: dict[str, list[str]] = {}
    for t in triplets:
        txts = load_texts(t, "test")
        test_embs[t] = enc.encode(txts, show_progress_bar=False, batch_size=64)
        test_texts[t] = txts

    rows = []
    for row in cross["results"]:
        m, e = row["model_triplet"], row["eval_triplet"]
        base = row["base"]["test"]
        unlearn = row["unlearn"]["test"]
        if not base or not unlearn or len(base) != len(unlearn):
            continue
        F = forget_embs[m]
        Fg = forget_geom[m]
        T = test_embs[e]

        # cosine + euclidean from target texts to forget set
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        sim = cosine_similarity(T, F)                       # (n_target, n_forget)
        dist = euclidean_distances(T, F)
        cent = Fg["centroid"]
        cent_unit = cent / max(np.linalg.norm(cent), 1e-12)

        for j in range(len(base)):
            b, u = base[j], unlearn[j]
            y = math.log(max(u["ppl"], 1e-6) / max(b["ppl"], 1e-6))
            tv = T[j]
            sims = sim[j]
            dists = dist[j]
            ss = np.sort(sims)[::-1]

            rows.append({
                "model_triplet": m,
                "eval_triplet": e,
                "sample_index": j,
                "same_cluster": int(m == e),
                # target ↔ forget cosine
                "cos_sim_to_centroid": float(
                    float(np.dot(tv, cent)) / (np.linalg.norm(tv) * np.linalg.norm(cent) + 1e-12)
                ),
                "cos_sim_to_nearest": float(sims.max()),
                "cos_sim_top3_mean": float(ss[:3].mean()),
                "cos_sim_top5_mean": float(ss[:5].mean()),
                "cos_sim_mean": float(sims.mean()),
                "cos_sim_std": float(sims.std()),
                # target ↔ forget euclidean
                "eucl_to_centroid": float(np.linalg.norm(tv - cent)),
                "eucl_to_nearest": float(dists.min()),
                # geometric projections
                "proj_on_centroid": float(np.dot(tv, cent_unit)),
                "angle_to_centroid_deg": float(
                    math.degrees(math.acos(max(min(
                        float(np.dot(tv, cent) /
                              (np.linalg.norm(tv) * np.linalg.norm(cent) + 1e-12)),
                        1.0), -1.0)))
                ),
                # forget intrinsic
                "forget_emb_variance_mean": Fg["emb_variance_mean"],
                "forget_mean_pairwise_similarity": Fg["mean_pairwise_similarity"],
                "forget_centroid_norm": Fg["centroid_norm"],
                "forget_spread": Fg["spread"],
                # target intrinsic
                "target_emb_norm": float(np.linalg.norm(tv)),
                # label
                "base_ppl": b["ppl"],
                "unlearn_ppl": u["ppl"],
                "log_ppl_ratio": y,
            })
    df = pd.DataFrame(rows)
    print(f"Built {len(df)} rows × {df.shape[1]} cols")
    return df


def evaluate_logo(X, y, groups, model):
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(y, dtype=float)
    for tr, te in logo.split(X, y, groups):
        m = type(model)(**model.get_params())
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    return y_pred


def report(name, y, yp):
    r2 = r2_score(y, yp)
    mae = mean_absolute_error(y, yp)
    rmse = math.sqrt(mean_squared_error(y, yp))
    print(f"  {name:<32s}  R²={r2:+.4f}   RMSE={rmse:.4f}   MAE={mae:.4f}")
    return {"r2": r2, "mae": mae, "rmse": rmse}


def main():
    csv_path = OUT_DIR / "corruption_geometry_features.csv"
    if csv_path.exists():
        print(f"Loading cached features from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = build_features()
        df.to_csv(csv_path, index=False)
        print(f"Wrote features to {csv_path}")

    id_cols = {"model_triplet", "eval_triplet", "sample_index"}
    label_cols = {"base_ppl", "unlearn_ppl", "log_ppl_ratio"}
    feature_cols = [c for c in df.columns if c not in id_cols | label_cols]
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].values.astype(float)
    y = df["log_ppl_ratio"].values.astype(float)
    groups = df["eval_triplet"].values

    # Mean baseline (predict the training mean).
    print("\n" + "=" * 72)
    print("  Baselines and models  (LOGO CV by eval_triplet, n=10 groups)")
    print("=" * 72)

    logo = LeaveOneGroupOut()
    y_mean = np.empty_like(y)
    for tr, te in logo.split(X, y, groups):
        y_mean[te] = y[tr].mean()
    mean_scores = report("mean-of-train baseline", y, y_mean)

    # Linear (Ridge) baseline on standardized features.
    X_std = StandardScaler().fit_transform(X)
    y_ridge = evaluate_logo(X_std, y, groups, Ridge(alpha=1.0))
    ridge_scores = report("Ridge regression", y, y_ridge)

    # Gradient boosting.
    y_gb = evaluate_logo(X, y, groups,
                        GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                  learning_rate=0.05, random_state=42))
    gb_scores = report("GradientBoosting (300/3/0.05)", y, y_gb)

    # Random forest.
    y_rf = evaluate_logo(X, y, groups,
                        RandomForestRegressor(n_estimators=500, max_depth=None,
                                              min_samples_leaf=3, max_features=0.5,
                                              random_state=42, n_jobs=-1))
    rf_scores = report("Random Forest (500/∞/3/0.5)", y, y_rf)

    # ── layer-conditional scores ────────────────────────────────────────────
    # Drop same-cluster rows and re-fit to see if geometry predicts *only spillover*.
    cross_mask = df["same_cluster"].values == 0
    Xc, yc, gc = X[cross_mask], y[cross_mask], groups[cross_mask]
    y_cross_rf = evaluate_logo(Xc, yc, gc,
                               RandomForestRegressor(n_estimators=500, max_depth=None,
                                                     min_samples_leaf=3, max_features=0.5,
                                                     random_state=42, n_jobs=-1))
    print("\n  -- restricted to cross-cluster rows only (L3 spillover) --")
    l3_scores = report("Random Forest on L3 only", yc, y_cross_rf)

    # ── feature importance from best RF fit on full data ───────────────────
    rf = RandomForestRegressor(n_estimators=500, max_depth=None,
                               min_samples_leaf=3, max_features=0.5,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = sorted(zip(feature_cols, rf.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 72)
    print("  RF feature importance  (full-data fit, all features are geometric)")
    print("=" * 72)
    for f, w in imp:
        bar = "#" * int(w * 200)
        print(f"  {f:<36s} {w:.4f}  {bar}")

    # ── per-layer residual analysis ─────────────────────────────────────────
    df_res = df.copy()
    df_res["y_pred"] = y_rf
    df_res["residual"] = df_res["log_ppl_ratio"] - df_res["y_pred"]
    print("\n" + "=" * 72)
    print("  Layer-conditioned LOGO-CV RF scores (full model, sliced)")
    print("=" * 72)
    for layer, mask in [
        ("L2 same-cluster (diag)", df_res["same_cluster"] == 1),
        ("L3 cross-cluster",       df_res["same_cluster"] == 0),
    ]:
        ys, ps = df_res.loc[mask, "log_ppl_ratio"].values, df_res.loc[mask, "y_pred"].values
        r2 = r2_score(ys, ps)
        mae = mean_absolute_error(ys, ps)
        rmse = math.sqrt(mean_squared_error(ys, ps))
        print(f"  {layer:<28s}  R²={r2:+.4f}   RMSE={rmse:.4f}   MAE={mae:.4f}   (n={mask.sum()})")

    # ── save ────────────────────────────────────────────────────────────────
    out_summary = {
        "n_rows": int(len(df)),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "scores": {
            "mean_baseline": mean_scores,
            "ridge": ridge_scores,
            "gradient_boosting": gb_scores,
            "random_forest_all": rf_scores,
            "random_forest_L3_only": l3_scores,
        },
        "feature_importance": {f: float(w) for f, w in imp},
    }
    with open(OUT_DIR / "geometry_results.json", "w") as f:
        json.dump(out_summary, f, indent=2)
    df_res.to_csv(OUT_DIR / "geometry_predictions.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'geometry_results.json'}")
    print(f"Wrote {OUT_DIR / 'geometry_predictions.csv'}")


if __name__ == "__main__":
    main()
