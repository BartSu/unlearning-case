"""
Extract per-text features for WikiText HDBSCAN triplets (test split).

~53 features per sample organised into:
  Text length            (6)  — token/char/word/sentence counts, avg word len, log length
  Text content          (15)  — word stats, entropy, capitalization, entities
  Text embedding PCA    (20)  — global PCA of text embeddings
  Text embedding        (10)  — norm, moments, centroid similarity
  Position               (2)  — normalized position, first-half flag
"""

import json
import re
import string
import warnings
from collections import Counter
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

WIKITEXT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data-preparation" / "data" / "wikitext_hdbscan_triplets"
)
OUT_DIR = Path(__file__).resolve().parent.parent
N_PCA = 20

_STOPWORDS = frozenset(
    "the a an and or but in on at to for of with by from is was are were be "
    "been being have has had do does did will would could should may might "
    "shall can it its this that these those he she they we you i me him her "
    "us them my your his their our what which who whom when where how not no "
    "nor if then than so as just also very too only about after before "
    "between into through during above below up down out off over under "
    "again further there here all each every both few more most other some "
    "such own same".split()
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(v, default=0.0):
    return v if np.isfinite(v) else default


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _sents(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()] or [text]


# ── per-sample feature builder ───────────────────────────────────────────────

def build_row(
    triplet_name: str,
    sample_idx: int,
    text: str,
    emb: np.ndarray,
    pca_coord: np.ndarray,
    centroid: np.ndarray,
    norm_rank: float,
    tokenizer,
    n_pca: int,
    test_set_size: int,
) -> dict:
    # ── text length (6) ──────────────────────────────────────────────────
    tok = len(tokenizer.encode(text, add_special_tokens=False))
    ws = _words(text)
    nw = len(ws) or 1
    wlens = [len(w) for w in ws]

    row: dict = {
        "split": triplet_name,
        "sample_index": sample_idx,
        "text_length_tokens": tok,
        "text_length_chars": len(text),
        "text_length_words": nw,
        "text_n_sentences": len(_sents(text)),
        "text_avg_word_length": float(np.mean(wlens)) if wlens else 0.0,
        "text_log_length": float(np.log1p(tok)),
    }

    # ── text content (15) ────────────────────────────────────────────────
    row["text_word_length_std"] = float(np.std(wlens)) if wlens else 0.0
    row["text_max_word_length"] = max(wlens) if wlens else 0
    row["text_unique_word_ratio"] = len(set(ws)) / nw
    row["text_char_diversity"] = len(set(text.lower())) / max(len(text), 1)
    row["text_stopword_ratio"] = sum(1 for w in ws if w in _STOPWORDS) / nw
    row["text_punctuation_count"] = sum(1 for c in text if c in string.punctuation)
    row["text_digit_count"] = sum(1 for c in text if c.isdigit())

    orig_w = text.split()
    cap = [w for w in orig_w if w[:1].isalpha() and w[0].isupper()]
    row["text_capitalized_count"] = len(cap)
    row["text_capitalized_ratio"] = len(cap) / max(len(orig_w), 1)

    ents: set[str] = set()
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        ents.add(m.group())
    for m in re.finditer(r"\b([A-Z]{2,})\b", text):
        ents.add(m.group())
    row["text_entity_count"] = len(ents)
    row["text_has_number"] = int(bool(re.search(r"\d", text)))

    if text:
        cc = Counter(text.lower())
        p = np.array([v / len(text) for v in cc.values()])
        row["text_char_entropy"] = float(-np.sum(p * np.log2(p + 1e-12)))
    else:
        row["text_char_entropy"] = 0.0
    if ws:
        wc = Counter(ws)
        p = np.array([v / nw for v in wc.values()])
        row["text_word_entropy"] = float(-np.sum(p * np.log2(p + 1e-12)))
    else:
        row["text_word_entropy"] = 0.0
    row["text_specificity"] = row["text_unique_word_ratio"] * nw

    # ── text embedding PCA (20) ──────────────────────────────────────────
    for d in range(n_pca):
        row[f"text_emb_pca_{d}"] = float(pca_coord[d])

    # ── text embedding stats (10) ────────────────────────────────────────
    t_norm = float(np.linalg.norm(emb))
    t_sim_cent = float(cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0, 0])
    row["text_emb_norm"] = t_norm
    row["text_emb_mean"] = float(emb.mean())
    row["text_emb_std"] = float(emb.std())
    row["text_emb_min"] = float(emb.min())
    row["text_emb_max"] = float(emb.max())
    row["text_emb_abs_mean"] = float(np.abs(emb).mean())
    row["text_sim_to_centroid"] = t_sim_cent
    row["text_dist_to_centroid"] = 1.0 - t_sim_cent
    row["text_emb_norm_pct"] = norm_rank
    row["text_emb_kurtosis"] = _safe(float(sp_stats.kurtosis(emb)))

    # ── position (2) ─────────────────────────────────────────────────────
    row["position_normalized"] = sample_idx / max(test_set_size, 1)
    row["is_first_half"] = int(sample_idx < test_set_size / 2)

    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract per-text features (~53 per sample) from test.json"
    )
    ap.add_argument("--data_dir", type=str, default=str(WIKITEXT_DIR))
    ap.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--tokenizer", type=str, default="bert-base-cased")
    ap.add_argument("--n_pca", type=int, default=N_PCA)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.output_dir)
    tdirs = sorted(data_dir.glob("triplet_*"))
    if args.end:
        tdirs = [d for d in tdirs if args.start <= int(d.name.split("_")[1]) <= args.end]
    else:
        tdirs = [d for d in tdirs if int(d.name.split("_")[1]) >= args.start]

    n = len(tdirs)
    print("=" * 72)
    print(f"  Text Feature Extraction (test.json): {n} triplets")
    print(f"  Embed model : {args.embed_model}")
    print(f"  PCA dims    : {args.n_pca}")
    print("=" * 72)

    print("Loading models ...")
    embed_model = SentenceTransformer(args.embed_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("  Ready.\n")

    # ── phase 1: collect all samples & encode ────────────────────────────
    all_texts: list[str] = []
    all_meta: list[tuple[str, int]] = []  # (triplet_name, local_index)
    triplet_sizes: dict[str, int] = {}
    for i, td in enumerate(tdirs, 1):
        tp = td / "test.json"
        if not tp.exists():
            print(f"  [{i:3d}/{n}] {td.name} — skip (no test.json)")
            continue
        with open(tp) as fh:
            samples = json.load(fh)
        triplet_sizes[td.name] = len(samples)
        for j, s in enumerate(samples):
            all_texts.append(s["text"])
            all_meta.append((td.name, j))
        print(f"  [{i:3d}/{n}] {td.name}  ({len(samples)} samples)")

    if not all_texts:
        print("No data found.")
        return

    print(f"\nEncoding {len(all_texts)} texts ...")
    embs = embed_model.encode(all_texts, show_progress_bar=True, batch_size=256)

    # ── phase 2: global PCA & centroid ───────────────────────────────────
    npca = min(args.n_pca, embs.shape[0], embs.shape[1])
    print(f"Fitting PCA ({embs.shape[0]} → {npca}) ...")
    pca = PCA(n_components=npca).fit(embs)
    pca_all = pca.transform(embs)
    print(f"  explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    centroid = embs.mean(axis=0)
    norms = np.linalg.norm(embs, axis=1)
    norm_ranks = sp_stats.rankdata(norms) / len(norms)

    # ── phase 3: per-sample rows ─────────────────────────────────────────
    print(f"\nBuilding {len(all_texts)} feature rows ...")
    rows: list[dict] = []
    for idx in range(len(all_texts)):
        triplet_name, local_idx = all_meta[idx]
        fs = triplet_sizes.get(triplet_name, 1)
        row = build_row(
            triplet_name, local_idx, all_texts[idx],
            embs[idx], pca_all[idx], centroid,
            float(norm_ranks[idx]), tokenizer, npca, fs,
        )
        rows.append(row)

    # ── output ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    n_feat = df.shape[1] - 2  # exclude split, sample_index

    csv_path = out_dir / "prompt_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  saved to {csv_path}  ({df.shape[0]} rows × {n_feat} features)")

    json_path = out_dir / "prompt_features.json"
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"JSON saved to {json_path}")

    print(f"\n── Columns ({n_feat} features) ──")
    for i, col in enumerate(df.columns, 1):
        if col not in ("split", "sample_index"):
            print(f"  {i:3d}. {col}")


if __name__ == "__main__":
    main()
