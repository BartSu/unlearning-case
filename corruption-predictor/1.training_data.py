"""
Build per-question training data for the corruption predictor (RF).

Combines three groups of features per question:
  A. Forgetting-set features  (from feature-engineering/wikitext_features.csv)
  B. Target-question features (token counts of question, answer, source text)
  C. Similarity features      (embedding similarity between question/text and
                                the forgetting set)

Label: CW (base correct -> unlearn wrong) = 1, else 0

Output: corruption-predictor/training_data.csv
"""

import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
FEATURES_CSV = ROOT.parent / "feature-engineering" / "wikitext_features.csv"
PAIRS_DIR = ROOT.parent / "extract-label" / "wikitext_qa_labels" / "pairs"
FORGET_DIR = ROOT.parent / "data-preparation" / "data" / "wikitext_hdbscan_triplets"
QA_DIR = ROOT.parent / "data-preparation" / "data" / "wikitext_hdbscan_triplets_qa"

FORGET_FEATURE_COLS = [
    "mean_distance_among_classes (MD)",
    "fishers_discriminant_ratio (FDR)",
    "calinski_harabasz_index (CHI)",
    "davies_bouldin_index (DBI)",
    "n_clusters (HDBSCAN)",
    "avg_n_tokens",
    "min_n_tokens",
    "max_n_tokens",
    "n_unique_tokens",
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_similarity_features(q_emb, text_emb, forget_embs, forget_mean):
    """Compute similarity features between a question/text and the forgetting set."""
    cos_q_all = cosine_similarity(q_emb, forget_embs)[0]
    cos_t_all = cosine_similarity(text_emb, forget_embs)[0]

    cos_q_mean = cosine_similarity(q_emb, forget_mean)[0, 0]
    cos_t_mean = cosine_similarity(text_emb, forget_mean)[0, 0]

    # percentile rank: how central is this source text within the forgetting set
    cos_t_self = cos_t_all.max()
    cos_t_all_sorted = np.sort(cosine_similarity(forget_embs, forget_mean).ravel())
    text_centrality_pct = float(np.searchsorted(cos_t_all_sorted, cos_t_mean)) / len(cos_t_all_sorted)

    return {
        "cos_q_to_forget_mean": round(float(cos_q_mean), 6),
        "cos_q_to_forget_max": round(float(cos_q_all.max()), 6),
        "cos_q_to_forget_std": round(float(cos_q_all.std()), 6),
        "cos_text_to_forget_mean": round(float(cos_t_mean), 6),
        "cos_text_to_forget_max": round(float(cos_t_self), 6),
        "cos_text_to_forget_std": round(float(cos_t_all.std()), 6),
        "text_centrality_pct": round(text_centrality_pct, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build training data: forgetting-set features + question features + similarity -> label"
    )
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--tokenizer", default="bert-base-cased")
    parser.add_argument("--output", default=str(ROOT / "training_data.csv"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feat_df = pd.read_csv(FEATURES_CSV)
    feat_lookup = {row["split"]: row for _, row in feat_df.iterrows()}

    pair_files = sorted(PAIRS_DIR.glob("*.json"))
    if not pair_files:
        print(f"No pair files in {PAIRS_DIR}")
        return

    print("Loading models ...")
    embed_model = SentenceTransformer(args.embed_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("  Ready.\n")

    rows = []
    for pf in pair_files:
        pair = load_json(pf)
        triplet = pair["eval_triplet"]

        if triplet not in feat_lookup:
            print(f"  Skipping {triplet}: no features in wikitext_features.csv")
            continue

        triplet_feats = feat_lookup[triplet]

        # Load and encode forgetting set
        forget_texts = [item["text"] for item in load_json(FORGET_DIR / triplet / "train.json")]
        forget_embs = embed_model.encode(forget_texts, show_progress_bar=False, batch_size=64)
        forget_mean = forget_embs.mean(axis=0, keepdims=True)

        # Load QA source data (for source text lookup)
        qa_data = load_json(QA_DIR / triplet / "train.json")
        qa_by_idx = {item["source_train_index"]: item for item in qa_data}

        # Gather questions and source texts
        questions, source_texts = [], []
        for ex in pair["examples"]:
            questions.append(ex["question"])
            src = qa_by_idx.get(ex["source_train_index"], {})
            source_texts.append(src.get("text", ""))

        q_embs = embed_model.encode(questions, show_progress_bar=False, batch_size=64)
        t_embs = embed_model.encode(source_texts, show_progress_bar=False, batch_size=64)

        n = len(pair["examples"])
        print(f"  {triplet}: {n} questions", end="")

        for i, ex in enumerate(pair["examples"]):
            sim_feats = compute_similarity_features(
                q_embs[i : i + 1], t_embs[i : i + 1], forget_embs, forget_mean
            )

            q_tok = len(tokenizer.encode(ex["question"], add_special_tokens=False))
            a_tok = len(tokenizer.encode(ex["answer"], add_special_tokens=False))
            text_tok = len(tokenizer.encode(source_texts[i], add_special_tokens=False)) if source_texts[i] else 0

            label = 1 if ex["case"] == "base_correct_unlearn_wrong" else 0

            row = {
                "triplet": triplet,
                "record_index": ex["record_index"],
                # A. Forgetting-set features
                **{col: triplet_feats[col] for col in FORGET_FEATURE_COLS},
                # B. Question features
                "q_n_tokens": q_tok,
                "a_n_tokens": a_tok,
                "text_n_tokens": text_tok,
                # C. Similarity features
                **sim_feats,
                # Meta
                "base_correct": ex["base_correct"],
                "unlearn_correct": ex["unlearn_correct"],
                "case": ex["case"],
                "label": label,
            }
            rows.append(row)

        n_cw = sum(1 for ex in pair["examples"] if ex["case"] == "base_correct_unlearn_wrong")
        print(f"  (CW={n_cw}, label=1 rate={n_cw/n:.2%})")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    feature_cols = FORGET_FEATURE_COLS + [
        "q_n_tokens", "a_n_tokens", "text_n_tokens",
        "cos_q_to_forget_mean", "cos_q_to_forget_max", "cos_q_to_forget_std",
        "cos_text_to_forget_mean", "cos_text_to_forget_max", "cos_text_to_forget_std",
        "text_centrality_pct",
    ]

    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    print(f"\n{'=' * 60}")
    print(f"  Training data saved to {output_path}")
    print(f"  Total samples:  {len(df)}")
    print(f"  Label=1 (CW):   {n_pos}  ({n_pos/len(df):.1%})")
    print(f"  Label=0:        {n_neg}  ({n_neg/len(df):.1%})")
    print(f"  Features:       {len(feature_cols)}")
    print(f"  Triplets:       {df['triplet'].nunique()}")
    print(f"{'=' * 60}")
    print(f"\n  Feature columns:")
    for c in feature_cols:
        print(f"    {c}")

    json_path = output_path.with_suffix(".json")
    meta = {
        "n_samples": len(df),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": round(n_pos / len(df), 4),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "triplets": sorted(df["triplet"].unique().tolist()),
        "embed_model": args.embed_model,
        "tokenizer": args.tokenizer,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  Metadata saved to {json_path}")


if __name__ == "__main__":
    main()
