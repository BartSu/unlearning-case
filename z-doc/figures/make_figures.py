"""Generate slide figures from audit artefacts.

Reads 4.regression-predictor/audit/*.csv + audit_summary.json and writes
three PDF figures into this directory, consumed by slides.tex / README-CN.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "4.regression-predictor" / "audit"
OUT = Path(__file__).resolve().parent

CLUSTER = {
    "triplet_001": "game",
    "triplet_011": "federer",
    "triplet_021": "jordan",
    "triplet_031": "episode",
    "triplet_041": "league",
    "triplet_051": "song",
    "triplet_061": "war",
    "triplet_071": "storm",
    "triplet_081": "river",
    "triplet_091": "star",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})


def fig_three_layer_decay() -> Path:
    summary = json.loads((AUDIT / "audit_summary.json").read_text())["layer_headline"]
    layers = ["L1_forget", "L2_locality", "L3_spillover"]
    ratios = [summary[k]["geo_mean_ratio"] for k in layers]
    pct10 = [summary[k]["pct_up_10"] for k in layers]
    ns = [summary[k]["n"] for k in layers]
    labels = [f"L1 forget\n(n={ns[0]})", f"L2 locality\n(n={ns[1]})", f"L3 spillover\n(n={ns[2]})"]

    fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(3)
    colors = ["#b2182b", "#ef8a62", "#fddbc7"]
    bars = ax1.bar(x, ratios, color=colors, edgecolor="black", width=0.6)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("geo-mean PPL ratio  $r = \\mathrm{PPL}_{\\mathrm{unl}}/\\mathrm{PPL}_{\\mathrm{base}}$")
    ax1.set_ylim(0.95, max(ratios) * 1.15)
    for b, v in zip(bars, ratios):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}×",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, pct10, "o-", color="#2166ac", linewidth=1.8, markersize=7, label="% samples with $r>1.1$")
    for xi, p in zip(x, pct10):
        ax2.text(xi, p - 5, f"{p:.1f}%", ha="center", va="top", color="#2166ac", fontsize=9)
    ax2.set_ylabel("fraction $r > 1.1$  (%)", color="#2166ac")
    ax2.tick_params(axis="y", colors="#2166ac")
    ax2.set_ylim(0, 110)
    ax2.spines["top"].set_visible(False)

    ax1.set_title("Three-layer PPL-ratio decay (pooled across 10 forget sets)")
    out = OUT / "fig_three_layer_decay.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_per_forget_profile() -> Path:
    df = pd.read_csv(AUDIT / "part1_corruption_profile.csv")
    df["cluster"] = df["forget_cluster"].map(CLUSTER)
    df = df.sort_values("geo_L1_forget").reset_index(drop=True)
    x = np.arange(len(df))
    w = 0.27

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w, df["geo_L1_forget"], w, color="#b2182b", label="L1 forget")
    ax.bar(x,     df["geo_L2_locality"], w, color="#ef8a62", label="L2 locality")
    ax.bar(x + w, df["geo_L3_spillover"], w, color="#fddbc7", edgecolor="black", label="L3 spillover")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["cluster"], rotation=30, ha="right")
    ax.set_ylabel("geo-mean PPL ratio")
    ax.set_title("Per–forget-set corruption profile (same unlearner, 10 forget sets)")
    ax.legend(loc="upper left", frameon=False)

    storm_idx = int(df.index[df["cluster"] == "storm"][0])
    ax.annotate("worst = storm", xy=(storm_idx, df.loc[storm_idx, "geo_L1_forget"]),
                xytext=(storm_idx - 2.2, df["geo_L1_forget"].max() * 1.03),
                arrowprops=dict(arrowstyle="->", color="#b2182b"),
                color="#b2182b", fontsize=9)
    out = OUT / "fig_per_forget_profile.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_audit_scatter() -> Path:
    pred = pd.read_csv(AUDIT / "part2_audit_predictions.csv")
    summary = json.loads((AUDIT / "audit_summary.json").read_text())["audit_predictor"]
    layers = [
        ("geo_L1_forget", "L1 forget", "#b2182b"),
        ("geo_L2_locality", "L2 locality", "#ef8a62"),
        ("geo_L3_spillover", "L3 spillover", "#4393c3"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    for ax, (key, label, color) in zip(axes, layers):
        t = pred[f"true_{key}"].to_numpy()
        p = pred[f"pred_{key}"].to_numpy()
        rho, _ = spearmanr(t, p)
        r2 = summary[key]["r2"]
        lo = min(t.min(), p.min()) * 0.98
        hi = max(t.max(), p.max()) * 1.02
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8)
        ax.scatter(t, p, color=color, s=45, edgecolor="black", linewidth=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("true geo-mean ratio")
        ax.set_title(f"{label}\n$R^2={r2:+.3f}$, $\\rho={rho:+.3f}$", fontsize=10)
    axes[0].set_ylabel("predicted (LOO, Ridge, 12-dim geometry)")
    fig.suptitle("Forget-set audit: no fine-tuning, n=10 LOO", y=1.02, fontsize=11)
    out = OUT / "fig_audit_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    paths = [fig_three_layer_decay(), fig_per_forget_profile(), fig_audit_scatter()]
    for p in paths:
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
