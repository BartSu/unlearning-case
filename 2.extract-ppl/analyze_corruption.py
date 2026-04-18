"""
Knowledge-corruption analysis from the cross-triplet PPL matrix.

Input:  wikitext_cross_metrics_detail.json  (10 unlearn ckpts × 10 eval triplets,
                                             each with per-sample base/unlearn
                                             loss & ppl on train/val/test splits)

Output: stdout tables + corruption_summary.json.

The matrix supports three distance-to-forget layers:
  L1  (forget)     : model_triplet == eval_triplet, split == "train"
                     → forgetting strength
  L2  (locality)   : model_triplet == eval_triplet, split in {"validation","test"}
                     → same-cluster collateral damage
  L3  (spillover)  : model_triplet != eval_triplet, split == "test"
                     → cross-cluster knowledge corruption

For each layer we report:
  - mean / median log(ppl_ratio)       (ratios are right-skewed; log is the
                                        right aggregation)
  - % samples with ratio > 1.1 and > 2 (how often surprisal actually rises)
  - per-cluster breakdown

We also report the diagonal vs off-diagonal ratio gap, which is the headline
locality number.
"""

from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

HERE = Path(__file__).resolve().parent
DETAIL = HERE / "wikitext_cross_metrics_detail.json"
OUT = HERE / "corruption_summary.json"


def log_ratio(base_ppl: float, unlearn_ppl: float) -> float:
    b = max(base_ppl, 1e-6)
    u = max(unlearn_ppl, 1e-6)
    return math.log(u / b)


def describe(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals)
    return {
        "n": int(a.size),
        "mean_log_ratio": float(a.mean()),
        "median_log_ratio": float(np.median(a)),
        "std_log_ratio": float(a.std()),
        "geo_mean_ratio": float(math.exp(a.mean())),
        "pct_up_10": float((a > math.log(1.1)).mean() * 100),
        "pct_up_2x": float((a > math.log(2.0)).mean() * 100),
        "pct_down_10": float((a < math.log(1 / 1.1)).mean() * 100),
    }


def main():
    with open(DETAIL) as fh:
        data = json.load(fh)

    # Pools of per-sample log-ratios by layer.
    L1, L2, L3 = [], [], []
    # Per-cluster (= eval_triplet) pools.
    by_cluster_L1 = defaultdict(list)
    by_cluster_L2 = defaultdict(list)
    by_cluster_L3_in  = defaultdict(list)  # ratios observed ON this cluster from other unlearners
    by_cluster_L3_out = defaultdict(list)  # ratios this cluster's unlearner caused ON others
    # Pair-level mean ratios for building a 10x10 log-ratio matrix on test split.
    pair_test_mean: dict[tuple[str, str], float] = {}

    for row in data["results"]:
        m, e = row["model_triplet"], row["eval_triplet"]
        base, unlearn = row["base"], row["unlearn"]

        # test split is always present
        test_logs = [
            log_ratio(b["ppl"], u["ppl"])
            for b, u in zip(base["test"], unlearn["test"])
        ]
        pair_test_mean[(m, e)] = float(np.mean(test_logs))

        if m == e:
            # L1: forget split
            for b, u in zip(base["train"], unlearn["train"]):
                lr = log_ratio(b["ppl"], u["ppl"])
                L1.append(lr)
                by_cluster_L1[e].append(lr)
            # L2: same cluster's retain (validation) + held-out (test)
            for split_name in ("validation", "test"):
                for b, u in zip(base[split_name], unlearn[split_name]):
                    lr = log_ratio(b["ppl"], u["ppl"])
                    L2.append(lr)
                    by_cluster_L2[e].append(lr)
        else:
            # L3: cross-cluster spillover, measured on test split
            for lr in test_logs:
                L3.append(lr)
                by_cluster_L3_in[e].append(lr)
                by_cluster_L3_out[m].append(lr)

    # ── top-line summary ────────────────────────────────────────────────────
    summary = {
        "L1_forget": describe(L1),
        "L2_locality_same_cluster": describe(L2),
        "L3_cross_cluster_spillover": describe(L3),
    }

    print("=" * 72)
    print("  Knowledge-corruption layers (per-sample log(ppl_ratio))")
    print("=" * 72)
    for key, d in summary.items():
        if d["n"] == 0:
            continue
        print(f"\n{key}  (n={d['n']})")
        print(f"  geo-mean ratio   : {d['geo_mean_ratio']:.3f}x")
        print(f"  mean  log-ratio  : {d['mean_log_ratio']:+.4f}")
        print(f"  median log-ratio : {d['median_log_ratio']:+.4f}")
        print(f"  std   log-ratio  : {d['std_log_ratio']:.4f}")
        print(f"  pct samples >1.1x: {d['pct_up_10']:5.1f}%")
        print(f"  pct samples >2.0x: {d['pct_up_2x']:5.1f}%")

    # ── headline: locality drop-off  L1 → L2 → L3 ──────────────────────────
    if L1 and L2 and L3:
        print("\n" + "=" * 72)
        print("  Locality decay (geo-mean ratio)")
        print("=" * 72)
        g1 = summary["L1_forget"]["geo_mean_ratio"]
        g2 = summary["L2_locality_same_cluster"]["geo_mean_ratio"]
        g3 = summary["L3_cross_cluster_spillover"]["geo_mean_ratio"]
        print(f"  L1 forget   : {g1:.3f}x")
        print(f"  L2 locality : {g2:.3f}x   (retained {(g2-1)/(g1-1)*100:5.1f}% of L1 damage)"
              if g1 > 1 else f"  L2 locality : {g2:.3f}x")
        print(f"  L3 spillover: {g3:.3f}x   (retained {(g3-1)/(g1-1)*100:5.1f}% of L1 damage)"
              if g1 > 1 else f"  L3 spillover: {g3:.3f}x")

    # ── per-cluster breakdown ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Per-cluster breakdown  (geo-mean ppl_ratio)")
    print("=" * 72)
    print(f"  {'cluster':<15} {'L1 forget':>10} {'L2 local':>10} "
          f"{'L3 in':>10} {'L3 out':>10}")
    cluster_stats = {}
    for c in sorted(by_cluster_L1):
        g1 = math.exp(mean(by_cluster_L1[c]))
        g2 = math.exp(mean(by_cluster_L2[c])) if by_cluster_L2[c] else float("nan")
        g3i = math.exp(mean(by_cluster_L3_in[c])) if by_cluster_L3_in[c] else float("nan")
        g3o = math.exp(mean(by_cluster_L3_out[c])) if by_cluster_L3_out[c] else float("nan")
        print(f"  {c:<15} {g1:>10.3f} {g2:>10.3f} {g3i:>10.3f} {g3o:>10.3f}")
        cluster_stats[c] = {
            "geo_L1_forget": g1,
            "geo_L2_locality": g2,
            "geo_L3_in": g3i,   # damage received from others
            "geo_L3_out": g3o,  # damage caused on others
        }
    summary["per_cluster"] = cluster_stats

    # ── 10x10 log-ratio matrix ──────────────────────────────────────────────
    clusters = sorted(by_cluster_L1)
    matrix = np.full((len(clusters), len(clusters)), np.nan)
    for i, m in enumerate(clusters):
        for j, e in enumerate(clusters):
            v = pair_test_mean.get((m, e))
            if v is not None:
                matrix[i, j] = v

    print("\n" + "=" * 72)
    print("  10x10 geo-mean ppl_ratio (model-row × eval-col, test split)")
    print("=" * 72)
    header = "  " + "".join(f"{c[-3:]:>8}" for c in clusters)
    print(header)
    for i, m in enumerate(clusters):
        row = "  ".join(
            f"{math.exp(matrix[i, j]):>6.2f}" if not math.isnan(matrix[i, j]) else "   nan"
            for j in range(len(clusters))
        )
        print(f"  {m[-3:]:>3}  " + row)

    # ── diagonal vs off-diagonal gap ────────────────────────────────────────
    diag = [matrix[i, i] for i in range(len(clusters)) if not math.isnan(matrix[i, i])]
    off = [matrix[i, j] for i in range(len(clusters)) for j in range(len(clusters))
           if i != j and not math.isnan(matrix[i, j])]
    if diag and off:
        print("\n" + "=" * 72)
        print("  Self vs cross on TEST split")
        print("=" * 72)
        print(f"  self (same-cluster test)  : geo={math.exp(np.mean(diag)):.3f}x  (n={len(diag)})")
        print(f"  cross (other-cluster test): geo={math.exp(np.mean(off)):.3f}x  (n={len(off)})")
        gap = math.exp(np.mean(diag)) / math.exp(np.mean(off))
        print(f"  locality multiplier       : {gap:.3f}x  "
              "(how much worse the unlearner hurts same-cluster text vs. other-cluster text)")
        summary["locality_multiplier_self_over_cross"] = gap
        summary["geo_self_test"]  = float(math.exp(np.mean(diag)))
        summary["geo_cross_test"] = float(math.exp(np.mean(off)))

    # ── correlation: cluster compactness ↔ corruption ─────────────────────
    try:
        import pandas as pd
        cf = pd.read_csv(HERE / "cluster_features.csv")
        cf["cluster_triplet"] = cf["triplet"]
        rows = []
        for c, s in cluster_stats.items():
            cf_row = cf[cf["cluster_triplet"] == c]
            if cf_row.empty:
                continue
            rows.append({
                "cluster": c,
                "embedding_variance": float(cf_row["embedding_variance"].iloc[0]),
                "mean_pairwise_similarity": float(cf_row["mean_pairwise_similarity"].iloc[0]),
                "token_entropy": float(cf_row["token_entropy"].iloc[0]),
                "geo_L1_forget":   s["geo_L1_forget"],
                "geo_L2_locality": s["geo_L2_locality"],
                "geo_L3_out":      s["geo_L3_out"],
            })
        df = pd.DataFrame(rows)
        if len(df) >= 3:
            print("\n" + "=" * 72)
            print("  Cluster geometry ↔ corruption (Pearson rho, n=%d)" % len(df))
            print("=" * 72)
            corr_table = {}
            for x in ("embedding_variance", "mean_pairwise_similarity", "token_entropy"):
                row = {}
                for y in ("geo_L1_forget", "geo_L2_locality", "geo_L3_out"):
                    r = df[[x, y]].corr().iloc[0, 1]
                    row[y] = float(r)
                corr_table[x] = row
                print(f"  {x:<28}  L1={row['geo_L1_forget']:+.2f}  "
                      f"L2={row['geo_L2_locality']:+.2f}  L3_out={row['geo_L3_out']:+.2f}")
            summary["geometry_corruption_correlations"] = corr_table
            summary["per_cluster_table"] = df.round(4).to_dict(orient="records")
    except Exception as err:
        print(f"\n(cluster_features.csv correlation skipped: {err})")

    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
