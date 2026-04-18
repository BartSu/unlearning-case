"""
Enriched experiments for the "forget-set audit" paper framing.

Narrative
---------
Part 1 (before vs. after unlearn) :
    Every forget set is characterized by a three-layer corruption
    profile — L1 forget / L2 same-cluster / L3 cross-cluster — measured
    by ppl_ratio = unlearn_ppl / base_ppl.

Part 2 (audit instead of unlearn) :
    Running unlearning + cross-evaluation to get that profile is expensive.
    Instead, we audit the forget set BEFORE unlearning, using only its
    own embedding geometry, and predict the three-layer profile.

Outputs (under 4.regression-predictor/audit/)
  - part1_corruption_profile.csv     (per-forget-set L1/L2/L3 geo-mean)
  - part1_per_sample_layers.csv      (every ppl_ratio with layer label)
  - part2_forget_features.csv        (forget-set intrinsic geometry)
  - part2_audit_predictions.csv      (loo predictions of L1, L2, L3)
  - part3_ranking.json               (Spearman ρ of predicted vs. true rank)
  - part3_coverage.csv               (retain-coverage by audit radius)
  - audit_summary.json               (final headline numbers)
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
TRIPLET_DIR = ROOT / "1.data-preparation" / "data" / "wikitext_hdbscan_triplets"
CROSS_JSON = ROOT / "2.extract-ppl" / "wikitext_cross_metrics_detail.json"
OUT = Path(__file__).resolve().parent / "audit"
OUT.mkdir(exist_ok=True, parents=True)


# ── helpers ─────────────────────────────────────────────────────────────────

def load_texts(tid: str, split: str) -> list[str]:
    with open(TRIPLET_DIR / tid / f"{split}.json") as f:
        return [x["text"] for x in json.load(f)]


def log_r(b: float, u: float) -> float:
    return math.log(max(u, 1e-6) / max(b, 1e-6))


def geo(a: np.ndarray) -> float:
    return float(math.exp(a.mean())) if len(a) else float("nan")


# ── Part 1: three-layer corruption ground truth ─────────────────────────────

def part1_ground_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(CROSS_JSON) as f:
        cross = json.load(f)

    rows, profile = [], {}
    for r in cross["results"]:
        m, e = r["model_triplet"], r["eval_triplet"]
        for split in ("train", "validation", "test"):
            base = r["base"].get(split, [])
            unl = r["unlearn"].get(split, [])
            if not base or not unl:
                continue
            for j, (b, u) in enumerate(zip(base, unl)):
                lr = log_r(b["ppl"], u["ppl"])
                if m == e and split == "train":
                    layer = "L1_forget"
                elif m == e and split in ("validation", "test"):
                    layer = "L2_locality"
                elif m != e and split == "test":
                    layer = "L3_spillover"
                else:
                    continue
                rows.append({
                    "forget_cluster": m, "eval_cluster": e, "split": split,
                    "sample_index": j, "layer": layer, "log_ppl_ratio": lr,
                })

    per_sample = pd.DataFrame(rows)

    # Per-forget-set profile: geo-mean ratio for each layer.
    profile_rows = []
    for fc in sorted(per_sample["forget_cluster"].unique()):
        sub = per_sample[per_sample["forget_cluster"] == fc]
        prof = {"forget_cluster": fc}
        for layer in ("L1_forget", "L2_locality", "L3_spillover"):
            vals = sub.loc[sub["layer"] == layer, "log_ppl_ratio"].values
            prof[f"geo_{layer}"] = geo(vals)
            prof[f"n_{layer}"] = int(len(vals))
        profile_rows.append(prof)
    profile_df = pd.DataFrame(profile_rows)
    return per_sample, profile_df


# ── Part 2: forget-set intrinsic geometry (audit features, no label) ───────

def part2_forget_features(clusters: list[str]) -> pd.DataFrame:
    print("Loading sentence-transformer ...")
    enc = SentenceTransformer("all-MiniLM-L6-v2")

    feats = []
    for fc in clusters:
        txts = load_texts(fc, "train")
        embs = enc.encode(txts, show_progress_bar=False, batch_size=64)
        centroid = embs.mean(axis=0)
        c_norm = float(np.linalg.norm(centroid))
        cos_mat = cosine_similarity(embs)
        euc_mat = euclidean_distances(embs)
        triu = np.triu_indices(len(embs), k=1)
        norms = np.linalg.norm(embs, axis=1)
        var = embs.var(axis=0)

        # PCA-free "effective rank"
        cov = np.cov(embs.T)
        eig = np.linalg.eigvalsh(cov)
        eig = np.clip(eig, 1e-12, None)
        p = eig / eig.sum()
        eff_rank = float(math.exp(-(p * np.log(p)).sum()))

        feats.append({
            "forget_cluster": fc,
            "emb_variance_mean":        float(var.mean()),
            "emb_variance_max":         float(var.max()),
            "pairwise_sim_mean":        float(cos_mat[triu].mean()),
            "pairwise_sim_std":         float(cos_mat[triu].std()),
            "pairwise_sim_q90":         float(np.percentile(cos_mat[triu], 90)),
            "pairwise_eucl_mean":       float(euc_mat[triu].mean()),
            "centroid_norm":            c_norm,
            "emb_norm_mean":            float(norms.mean()),
            "emb_norm_std":             float(norms.std()),
            "effective_rank":           eff_rank,
            "isotropy":                 float(var.min() / max(var.max(), 1e-12)),
            "spread_over_centroid":     float(euc_mat[triu].mean()) / max(c_norm, 1e-12),
        })
    return pd.DataFrame(feats)


# ── Part 2 predictor: regress L1/L2/L3 from forget-set-only features ───────

def part2_audit_predictor(prof: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    df = feat.merge(prof, on="forget_cluster", how="inner").sort_values("forget_cluster")
    feature_cols = [c for c in feat.columns if c != "forget_cluster"]
    X = df[feature_cols].values.astype(float)

    # standardize for Ridge
    X_mean, X_std = X.mean(0), X.std(0) + 1e-12
    Xs = (X - X_mean) / X_std

    preds = df[["forget_cluster"]].copy()
    loo = LeaveOneOut()

    print("\n" + "=" * 72)
    print("  Part 2: forget-set-level audit predictor (LOO over 10 clusters)")
    print("=" * 72)

    for target in ("geo_L1_forget", "geo_L2_locality", "geo_L3_spillover"):
        y = df[target].values.astype(float)
        yhat = np.empty_like(y)
        for tr, te in loo.split(Xs):
            m = Ridge(alpha=1.0).fit(Xs[tr], y[tr])
            yhat[te] = m.predict(Xs[te])
        preds[f"pred_{target}"] = yhat
        preds[f"true_{target}"] = y
        r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rmse = float(np.sqrt(((y - yhat) ** 2).mean()))
        rho, p = spearmanr(y, yhat)
        pear, _ = pearsonr(y, yhat)
        print(f"  {target:<22s}  R²={r2:+.3f}   RMSE={rmse:.4f}   "
              f"ρ(Spearman)={rho:+.3f}  r(Pearson)={pear:+.3f}")
    return preds


# ── Part 3: ranking + coverage ──────────────────────────────────────────────

def part3_ranking(preds: pd.DataFrame) -> dict:
    out = {}
    for layer in ("L1_forget", "L2_locality", "L3_spillover"):
        y_true = preds[f"true_geo_{layer}"].values
        y_pred = preds[f"pred_geo_{layer}"].values
        true_rank = np.argsort(-y_true).argsort()
        pred_rank = np.argsort(-y_pred).argsort()
        rho, p = spearmanr(true_rank, pred_rank)
        top1_hit = int(np.argmax(y_true) == np.argmax(y_pred))
        top3_overlap = len(set(np.argsort(-y_true)[:3]) & set(np.argsort(-y_pred)[:3]))
        out[layer] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "top1_match": top1_hit,
            "top3_overlap": top3_overlap,
        }
    return out


def part3_coverage(feat: pd.DataFrame) -> pd.DataFrame:
    """For each forget set, compute fraction of retain texts (other clusters'
    validation+test) that fall within its 'high-risk' radius in embedding
    space. Radius = 95th percentile of intra-forget pairwise euclidean dist."""
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    clusters = feat["forget_cluster"].tolist()

    forget_embs, forget_radius = {}, {}
    for fc in clusters:
        txts = load_texts(fc, "train")
        embs = enc.encode(txts, show_progress_bar=False, batch_size=64)
        forget_embs[fc] = embs
        euc = euclidean_distances(embs)
        triu = np.triu_indices(len(embs), k=1)
        forget_radius[fc] = float(np.percentile(euc[triu], 95))

    retain_embs = {}
    for ec in clusters:
        txts = load_texts(ec, "validation") + load_texts(ec, "test")
        retain_embs[ec] = enc.encode(txts, show_progress_bar=False, batch_size=64)

    rows = []
    for fc in clusters:
        F = forget_embs[fc]
        R = forget_radius[fc]
        for ec in clusters:
            if fc == ec:
                continue
            T = retain_embs[ec]
            d = euclidean_distances(T, F).min(axis=1)
            cov = float((d <= R).mean())
            rows.append({
                "forget_cluster": fc, "retain_cluster": ec,
                "forget_radius_p95": R,
                "frac_within_radius": cov,
                "mean_min_dist": float(d.mean()),
            })
    return pd.DataFrame(rows)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Part 1 — Three-layer corruption ground truth (ppl_ratio-based)")
    print("=" * 72)
    per_sample, profile = part1_ground_truth()
    per_sample.to_csv(OUT / "part1_per_sample_layers.csv", index=False)
    profile.to_csv(OUT / "part1_corruption_profile.csv", index=False)

    # per-forget profile table
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 140)
    print(profile.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # layer-level headline numbers
    head = {}
    for layer in ("L1_forget", "L2_locality", "L3_spillover"):
        v = per_sample.loc[per_sample["layer"] == layer, "log_ppl_ratio"].values
        head[layer] = {
            "n": int(len(v)),
            "geo_mean_ratio":       geo(v),
            "mean_log":             float(v.mean()),
            "pct_up_10":            float((v > math.log(1.1)).mean() * 100),
            "pct_up_2x":            float((v > math.log(2.0)).mean() * 100),
        }
    print("\nLayer headline (from full per-sample pool):")
    for k, d in head.items():
        print(f"  {k:<14s}  geo={d['geo_mean_ratio']:.3f}x  "
              f">1.1x={d['pct_up_10']:.1f}%  >2x={d['pct_up_2x']:.1f}%  (n={d['n']})")

    print("\n" + "=" * 72)
    print("  Part 2 — Forget-set audit features")
    print("=" * 72)
    feat = part2_forget_features(profile["forget_cluster"].tolist())
    feat.to_csv(OUT / "part2_forget_features.csv", index=False)
    print(feat.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    preds = part2_audit_predictor(profile, feat)
    preds.to_csv(OUT / "part2_audit_predictions.csv", index=False)

    print("\n" + "=" * 72)
    print("  Part 3 — Ranking validation  (predicted vs. true rank, 10 clusters)")
    print("=" * 72)
    ranking = part3_ranking(preds)
    for layer, d in ranking.items():
        print(f"  {layer:<14s}  Spearman ρ = {d['spearman_rho']:+.3f}   "
              f"top-1 match = {bool(d['top1_match'])}   "
              f"top-3 overlap = {d['top3_overlap']}/3")

    print("\n" + "=" * 72)
    print("  Part 3b — Retain-coverage inside forget-radius (p95 euclid)")
    print("=" * 72)
    cov = part3_coverage(feat)
    cov.to_csv(OUT / "part3_coverage.csv", index=False)
    tab = cov.groupby("forget_cluster").agg(
        mean_coverage=("frac_within_radius", "mean"),
        max_coverage =("frac_within_radius", "max"),
        radius_p95   =("forget_radius_p95", "first"),
    ).reset_index()
    # merge in L3_out to show coverage ↔ spillover relationship
    tab = tab.merge(
        profile[["forget_cluster", "geo_L3_spillover"]],
        on="forget_cluster",
    )
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    cov_rho, _ = spearmanr(tab["mean_coverage"], tab["geo_L3_spillover"])
    cov_pear, _ = pearsonr(tab["mean_coverage"], tab["geo_L3_spillover"])
    print(f"\n  mean retain-coverage  vs  actual L3_spillover :  "
          f"ρ={cov_rho:+.3f}  r={cov_pear:+.3f}")

    summary = {
        "layer_headline": head,
        "audit_predictor": {
            k: {
                "r2": float(1 - ((preds[f"true_{k}"] - preds[f"pred_{k}"])**2).sum() /
                                 ((preds[f"true_{k}"] - preds[f"true_{k}"].mean())**2).sum()),
                "rmse": float(np.sqrt(((preds[f"true_{k}"] - preds[f"pred_{k}"])**2).mean())),
                "spearman_rho": ranking[k.replace("geo_", "")]["spearman_rho"],
                "top1_match": ranking[k.replace("geo_", "")]["top1_match"],
                "top3_overlap": ranking[k.replace("geo_", "")]["top3_overlap"],
            }
            for k in ("geo_L1_forget", "geo_L2_locality", "geo_L3_spillover")
        },
        "coverage_vs_spillover": {
            "spearman_rho": float(cov_rho),
            "pearson_r":    float(cov_pear),
        },
    }
    with open(OUT / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT}/audit_summary.json")


if __name__ == "__main__":
    main()
