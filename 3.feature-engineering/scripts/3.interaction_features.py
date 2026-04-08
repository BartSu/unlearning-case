"""
Extract test-text ↔ train-set interaction features for WikiText HDBSCAN triplets.

For each test sample, compute its relationship to the entire train set (forget set).

~74 features per sample organised into:
  Cosine similarity    (20) — centroid/nearest/distribution stats, thresholds
  Euclidean distance   (14) — centroid/nearest/distribution stats
  Entity overlap        (8) — count, ratio, Jaccard, Dice, containment
  Keyword overlap       (9) — count, ratio, Jaccard, novelty, coverage
  N-gram overlap        (9) — word/char bigram-4gram, BLEU, ROUGE-L
  Rank / position       (8) — nearest rank, Gini, entropy, gaps
  Cross-embedding       (6) — test-text vs centroid geometry
"""

import json
import re
import warnings
from collections import Counter
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

WIKITEXT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data-preparation" / "data" / "wikitext_hdbscan_triplets"
)
OUT_DIR = Path(__file__).resolve().parent.parent

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


# ── text helpers ─────────────────────────────────────────────────────────────

def _safe(v, default=0.0):
    return v if np.isfinite(v) else default


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _extract_entities(text: str) -> set[str]:
    ents: set[str] = set()
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        ents.add(m.group().lower())
    for m in re.finditer(r"\b([A-Z]{2,})\b", text):
        ents.add(m.group().lower())
    return ents


def _extract_keywords(text: str) -> set[str]:
    return {t for t in _words(text) if t not in _STOPWORDS and len(t) > 2}


def _word_ngrams(words: list[str], n: int) -> set[tuple]:
    return set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def _char_ngrams(text: str, n: int) -> set[str]:
    t = text.lower()
    return set(t[i : i + n] for i in range(len(t) - n + 1))


def _lcs_length(x: list, y: list) -> int:
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _safe_div(a, b):
    return a / b if b else 0.0


# ── precomputation per triplet (aggregated train set) ────────────────────────

def _precompute_train(train_texts: list[str]):
    all_text = " ".join(train_texts)
    all_ws = _words(all_text)
    return {
        "entities": _extract_entities(all_text),
        "keywords": _extract_keywords(all_text),
        "word_set": set(all_ws),
        "word_bigrams": _word_ngrams(all_ws, 2),
        "word_trigrams": _word_ngrams(all_ws, 3),
        "char_bigrams": _char_ngrams(all_text, 2),
        "char_trigrams": _char_ngrams(all_text, 3),
        "char_4grams": _char_ngrams(all_text, 4),
        "word_counter": Counter(all_ws),
    }


# ── per-sample interaction features ─────────────────────────────────────────

def compute_interaction(
    test_emb: np.ndarray,
    train_embs: np.ndarray,
    test_text: str,
    train_texts: list[str],
    pre: dict,
) -> dict:
    n_t = train_embs.shape[0]
    centroid = train_embs.mean(axis=0)

    # ── cosine similarity (20) ───────────────────────────────────────────
    sims = cosine_similarity(test_emb.reshape(1, -1), train_embs).ravel()
    sim_cent = float(cosine_similarity(test_emb.reshape(1, -1), centroid.reshape(1, -1))[0, 0])
    ss = np.sort(sims)[::-1]
    sim_std = float(sims.std())

    nearest_idx = int(np.argmax(sims))

    f: dict = {
        "cos_sim_to_centroid": sim_cent,
        "cos_sim_to_nearest": float(sims[nearest_idx]),
        "cos_sim_max": float(sims.max()),
        "cos_sim_min": float(sims.min()),
        "cos_sim_mean": float(sims.mean()),
        "cos_sim_std": sim_std,
        "cos_sim_median": float(np.median(sims)),
        "cos_sim_skew": _safe(float(sp_stats.skew(sims))) if n_t >= 3 and sim_std > 1e-9 else 0.0,
        "cos_sim_kurtosis": _safe(float(sp_stats.kurtosis(sims))) if n_t >= 4 and sim_std > 1e-9 else 0.0,
        "cos_sim_q10": float(np.percentile(sims, 10)),
        "cos_sim_q25": float(np.percentile(sims, 25)),
        "cos_sim_q75": float(np.percentile(sims, 75)),
        "cos_sim_q90": float(np.percentile(sims, 90)),
        "cos_sim_top3_avg": float(ss[: min(3, n_t)].mean()),
        "cos_sim_top5_avg": float(ss[: min(5, n_t)].mean()),
        "cos_sim_top10_avg": float(ss[: min(10, n_t)].mean()),
        "cos_sim_above_05": int((sims > 0.5).sum()),
        "cos_sim_above_07": int((sims > 0.7).sum()),
        "cos_sim_above_09": int((sims > 0.9).sum()),
        "cos_sim_cv": float(sim_std / max(abs(sims.mean()), 1e-12)),
    }

    # ── euclidean distance (14) ──────────────────────────────────────────
    dists = euclidean_distances(test_emb.reshape(1, -1), train_embs).ravel()
    d_cent = float(euclidean_distances(test_emb.reshape(1, -1), centroid.reshape(1, -1))[0, 0])
    d_std = float(dists.std())
    f.update({
        "eucl_to_centroid": d_cent,
        "eucl_to_nearest": float(dists.min()),
        "eucl_min": float(dists.min()),
        "eucl_max": float(dists.max()),
        "eucl_mean": float(dists.mean()),
        "eucl_std": d_std,
        "eucl_median": float(np.median(dists)),
        "eucl_q10": float(np.percentile(dists, 10)),
        "eucl_q25": float(np.percentile(dists, 25)),
        "eucl_q75": float(np.percentile(dists, 75)),
        "eucl_q90": float(np.percentile(dists, 90)),
        "eucl_range": float(dists.max() - dists.min()),
        "eucl_iqr": float(np.percentile(dists, 75) - np.percentile(dists, 25)),
        "eucl_cv": float(d_std / max(abs(dists.mean()), 1e-12)),
    })

    # ── entity overlap (8) ───────────────────────────────────────────────
    test_ent = _extract_entities(test_text)
    train_ent = pre["entities"]
    e_inter = test_ent & train_ent
    te_n = len(test_ent) or 1
    tr_en = len(train_ent) or 1
    e_union = test_ent | train_ent

    f.update({
        "entity_overlap_count": len(e_inter),
        "entity_overlap_ratio_test": _safe_div(len(e_inter), te_n),
        "entity_overlap_ratio_train": _safe_div(len(e_inter), tr_en),
        "entity_jaccard": _safe_div(len(e_inter), max(len(e_union), 1)),
        "entity_dice": _safe_div(2 * len(e_inter), te_n + tr_en),
        "entity_containment_in_train": _safe_div(len(e_inter), te_n),
        "test_entity_count": len(test_ent),
        "train_entity_count": len(train_ent),
    })

    # ── keyword overlap (9) ──────────────────────────────────────────────
    test_kw = _extract_keywords(test_text)
    train_kw = pre["keywords"]
    k_inter = test_kw & train_kw
    tkn = len(test_kw) or 1
    trkn = len(train_kw) or 1
    k_union = test_kw | train_kw

    f.update({
        "keyword_overlap_count": len(k_inter),
        "keyword_overlap_ratio": _safe_div(len(k_inter), tkn),
        "keyword_jaccard": _safe_div(len(k_inter), max(len(k_union), 1)),
        "keyword_dice": _safe_div(2 * len(k_inter), tkn + trkn),
        "keyword_containment_in_train": _safe_div(len(k_inter), tkn),
        "test_keyword_count": len(test_kw),
        "train_keyword_count": len(train_kw),
        "keyword_novelty": _safe_div(len(test_kw - train_kw), tkn),
        "keyword_coverage": _safe_div(len(k_inter), trkn),
    })

    # ── n-gram overlap (9) ───────────────────────────────────────────────
    test_ws = _words(test_text)
    tbi = _word_ngrams(test_ws, 2)
    ttri = _word_ngrams(test_ws, 3)
    tr_bi = pre["word_bigrams"]
    tr_tri = pre["word_trigrams"]

    f["word_bigram_overlap"] = len(tbi & tr_bi)
    f["word_bigram_jaccard"] = _safe_div(len(tbi & tr_bi), max(len(tbi | tr_bi), 1))
    f["word_trigram_overlap"] = len(ttri & tr_tri)
    f["word_trigram_jaccard"] = _safe_div(len(ttri & tr_tri), max(len(ttri | tr_tri), 1))

    tc2 = _char_ngrams(test_text, 2)
    tc3 = _char_ngrams(test_text, 3)
    tc4 = _char_ngrams(test_text, 4)
    f["char_bigram_overlap"] = _safe_div(len(tc2 & pre["char_bigrams"]), max(len(tc2), 1))
    f["char_trigram_overlap"] = _safe_div(len(tc3 & pre["char_trigrams"]), max(len(tc3), 1))
    f["char_4gram_overlap"] = _safe_div(len(tc4 & pre["char_4grams"]), max(len(tc4), 1))

    test_uni = Counter(test_ws)
    train_uni = pre["word_counter"]
    f["bleu_1"] = _safe_div(sum((test_uni & train_uni).values()), max(sum(test_uni.values()), 1))

    nearest_ws = _words(train_texts[nearest_idx])
    lcs = _lcs_length(test_ws, nearest_ws)
    f["rouge_l_recall_nearest"] = _safe_div(lcs, max(len(test_ws), 1))

    # ── rank / position (8) ──────────────────────────────────────────────
    sa = np.argsort(sims)[::-1]
    f["nearest_train_idx"] = int(sa[0])
    f["sim_nearest_vs_mean"] = float(sims[sa[0]]) - float(sims.mean())
    f["frac_above_mean_sim"] = float((sims > sims.mean()).sum()) / n_t

    s_sorted = np.sort(np.abs(sims))
    idx_arr = np.arange(1, n_t + 1)
    s_sum = s_sorted.sum()
    f["sim_gini"] = float(
        (2 * (idx_arr * s_sorted).sum() / (n_t * s_sum)) - (n_t + 1) / n_t
    ) if s_sum > 1e-12 else 0.0

    p = np.abs(sims)
    ps = p.sum()
    if ps > 1e-12:
        p = p / ps
        f["sim_entropy"] = float(-np.sum(p * np.log2(p + 1e-12)))
    else:
        f["sim_entropy"] = 0.0
    f["sim_gap_top12"] = float(ss[0] - ss[1]) if n_t >= 2 else 0.0
    f["sim_gap_top23"] = float(ss[1] - ss[2]) if n_t >= 3 else 0.0
    f["sim_concentration_top5"] = float(ss[:min(5, n_t)].sum() / max(s_sum, 1e-12))

    # ── cross-embedding (6) ──────────────────────────────────────────────
    t_norm = float(np.linalg.norm(test_emb))
    c_norm = float(np.linalg.norm(centroid))
    c_unit = centroid / max(c_norm, 1e-12)

    f["test_proj_centroid"] = float(np.dot(test_emb, c_unit))
    f["test_centroid_angle"] = float(np.degrees(np.arccos(np.clip(sim_cent, -1, 1))))
    f["test_emb_norm"] = t_norm
    f["centroid_norm"] = c_norm
    f["test_norm_vs_centroid_norm"] = t_norm / max(c_norm, 1e-12)
    f["test_centroid_dist"] = d_cent

    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in f.items()}


# ── per-triplet extraction ──────────────────────────────────────────────────

def extract_triplet(name, train_texts, test_texts, embed_model):
    train_embs = embed_model.encode(train_texts, show_progress_bar=False, batch_size=64)
    test_embs = embed_model.encode(test_texts, show_progress_bar=False, batch_size=64)

    pre = _precompute_train(train_texts)

    rows: list[dict] = []
    for j, test_text in enumerate(test_texts):
        feats = compute_interaction(
            test_embs[j], train_embs, test_text, train_texts, pre,
        )
        row = {"split": name, "test_index": j}
        row.update(feats)
        rows.append(row)
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract test-text ↔ train-set interaction features (~74 per sample)"
    )
    ap.add_argument("--data_dir", type=str, default=str(WIKITEXT_DIR))
    ap.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
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
    print(f"  Interaction Feature Extraction (test ↔ train): {n} triplets")
    print(f"  Embed model : {args.embed_model}")
    print("=" * 72)

    print("Loading embedding model ...")
    embed_model = SentenceTransformer(args.embed_model)
    print("  Ready.\n")

    all_rows: list[dict] = []
    for i, td in enumerate(tdirs, 1):
        train_p = td / "train.json"
        test_p = td / "test.json"
        if not train_p.exists() or not test_p.exists():
            print(f"  [{i:3d}/{n}] {td.name} — skip (missing train/test)")
            continue
        with open(train_p) as fh:
            train_samples = json.load(fh)
        with open(test_p) as fh:
            test_samples = json.load(fh)
        train_texts = [s["text"] for s in train_samples]
        test_texts = [s["text"] for s in test_samples]
        print(
            f"  [{i:3d}/{n}] {td.name} (train={len(train_texts)}, test={len(test_texts)}) ...",
            end=" ", flush=True,
        )
        rows = extract_triplet(td.name, train_texts, test_texts, embed_model)
        all_rows.extend(rows)
        avg_csim = np.mean([r["cos_sim_to_centroid"] for r in rows])
        avg_ent = np.mean([r["entity_overlap_count"] for r in rows])
        print(f"sim_cent={avg_csim:.4f}  ent_overlap={avg_ent:.1f}")

    df = pd.DataFrame(all_rows)
    n_feat = df.shape[1] - 2  # exclude split, test_index

    csv_path = out_dir / "interaction_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  saved to {csv_path}  ({df.shape[0]} rows × {n_feat} features)")

    json_path = out_dir / "interaction_features.json"
    with open(json_path, "w") as fh:
        json.dump(all_rows, fh, indent=2, ensure_ascii=False)
    print(f"JSON saved to {json_path}")

    print(f"\n── Per-triplet summary ──")
    num_cols = [c for c in df.columns if c not in ("split", "test_index")]
    summary = df.groupby("split")[num_cols[:8]].agg(["mean"]).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
