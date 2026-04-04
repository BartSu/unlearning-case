"""
Extract dataset-level features for WikiText HDBSCAN triplets.

Features (per Dang et al., ACL 2024):
  Embedding Distribution:  MD, FDR, CHI, DBI, #clusters (HDBSCAN)
  Token-Based Statistics:  avg_tokens, min_tokens, max_tokens, #unique_tokens
"""

import json
import warnings
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import hdbscan as hdb
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

WIKITEXT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data-preparation" / "data" / "wikitext_hdbscan_triplets"
)
OUT_DIR = Path(__file__).resolve().parent.parent


# ── Feature Functions ────────────────────────────────────────────────────────

def mean_distance_among_classes(embeddings, labels):
    unique = sorted(set(labels))
    centroids = np.array([embeddings[np.array(labels) == c].mean(axis=0) for c in unique])
    k = len(centroids)
    if k < 2:
        return 0.0
    total = sum(
        (centroids[i] - centroids[j]) @ (centroids[i] - centroids[j])
        for i in range(k) for j in range(i + 1, k)
    )
    return 2.0 * total / (k * (k - 1))


def fisher_discriminant_ratio(embeddings, labels):
    labels_arr = np.array(labels)
    global_mean = embeddings.mean(axis=0)
    s_w, s_b = 0.0, 0.0
    for c in sorted(set(labels)):
        cluster = embeddings[labels_arr == c]
        centroid = cluster.mean(axis=0)
        s_w += ((cluster - centroid) ** 2).sum()
        diff_b = centroid - global_mean
        s_b += cluster.shape[0] * (diff_b @ diff_b)
    return s_b / s_w if s_w > 0 else 0.0


def calinski_harabasz_index(embeddings, labels):
    if len(set(labels)) < 2:
        return 0.0
    return calinski_harabasz_score(embeddings, labels)


def clustering_and_pseudo_labels(embeddings, min_cluster_size=5):
    """Run HDBSCAN; return (DBI, n_clusters, pseudo_labels).

    Pseudo-labels: valid cluster IDs for clustered points, -1 for noise.
    For features that need labels, noise points form their own class.
    """
    clusterer = hdb.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    valid = labels >= 0
    n_clusters = len(set(labels[valid])) if valid.any() else 0
    if n_clusters < 2:
        dbi = 0.0
    else:
        dbi = davies_bouldin_score(embeddings[valid], labels[valid])
    return dbi, max(n_clusters, 1), labels.tolist()


def token_statistics(all_ids, lengths):
    unique_tokens = len(set(tid for ids in all_ids for tid in ids))
    return float(lengths.mean()), int(lengths.min()), int(lengths.max()), unique_tokens


# ── Per-triplet Extraction ───────────────────────────────────────────────────

def extract_features_for_triplet(name, texts, embed_model, tokenizer):
    embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=64)

    all_ids, lengths = [], []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        all_ids.append(ids)
        lengths.append(len(ids))
    lengths = np.array(lengths)

    dbi, n_clust, pseudo_labels = clustering_and_pseudo_labels(embeddings)

    valid_mask = np.array(pseudo_labels) >= 0
    if valid_mask.sum() >= 2 and len(set(np.array(pseudo_labels)[valid_mask])) >= 2:
        emb_valid = embeddings[valid_mask]
        lab_valid = np.array(pseudo_labels)[valid_mask].tolist()
    else:
        emb_valid = embeddings
        lab_valid = pseudo_labels

    md = mean_distance_among_classes(emb_valid, lab_valid)
    fdr = fisher_discriminant_ratio(emb_valid, lab_valid)
    chi = calinski_harabasz_index(emb_valid, lab_valid)
    avg_tok, min_tok, max_tok, uniq_tok = token_statistics(all_ids, lengths)

    return {
        "split": name,
        "n_samples": len(texts),
        "mean_distance_among_classes (MD)": round(float(md), 6),
        "fishers_discriminant_ratio (FDR)": round(float(fdr), 6),
        "calinski_harabasz_index (CHI)": round(float(chi), 6),
        "davies_bouldin_index (DBI)": round(float(dbi), 6),
        "n_clusters (HDBSCAN)": int(n_clust),
        "avg_n_tokens": round(avg_tok, 2),
        "min_n_tokens": min_tok,
        "max_n_tokens": max_tok,
        "n_unique_tokens": uniq_tok,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract Embedding Distribution + Token Statistics for WikiText triplets"
    )
    parser.add_argument("--data_dir", type=str, default=str(WIKITEXT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--tokenizer", type=str, default="bert-base-cased")
    parser.add_argument("--start", type=int, default=1, help="First triplet index")
    parser.add_argument("--end", type=int, default=None, help="Last triplet index")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    triplet_dirs = sorted(data_dir.glob("triplet_*"))
    if args.end:
        triplet_dirs = [d for d in triplet_dirs
                        if args.start <= int(d.name.split("_")[1]) <= args.end]
    else:
        triplet_dirs = [d for d in triplet_dirs
                        if int(d.name.split("_")[1]) >= args.start]

    n = len(triplet_dirs)
    print("=" * 64)
    print(f"  WikiText Feature Extraction (Embed Dist + Token Stats): {n} triplets")
    print(f"  Embed model : {args.embed_model}")
    print(f"  Tokenizer   : {args.tokenizer}")
    print("=" * 64)

    print("Loading models (one-time cost) ...")
    embed_model = SentenceTransformer(args.embed_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("  Ready.\n")

    all_features = []
    for i, tdir in enumerate(triplet_dirs, 1):
        name = tdir.name
        with open(tdir / "train.json") as f:
            texts = [item["text"] for item in json.load(f)]

        print(f"  [{i:3d}/{n}] {name} ({len(texts)} texts) ...", end=" ", flush=True)
        feat = extract_features_for_triplet(name, texts, embed_model, tokenizer)
        all_features.append(feat)
        print(f"MD={feat['mean_distance_among_classes (MD)']:.4f}  "
              f"n_clust={feat['n_clusters (HDBSCAN)']:3d}  "
              f"avg_tok={feat['avg_n_tokens']:.1f}")

    df = pd.DataFrame(all_features)

    csv_path = output_dir / "wikitext_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  saved to {csv_path}")

    json_path = output_dir / "wikitext_features.json"
    with open(json_path, "w") as f:
        json.dump(all_features, f, indent=2)
    print(f"JSON saved to {json_path}")

    print(f"\n{'=' * 80}")
    print(df.to_string(index=False))
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
