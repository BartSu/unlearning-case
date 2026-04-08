"""
Extract forget-set level features for WikiText HDBSCAN triplets.

~147 features per triplet organised into:
  Length distributions  (52) — 4 distributions × 13 stats each
  Lexical              (15) — vocabulary diversity, word patterns
  Embedding geometry   (26) — norms, variance, PCA, isotropy
  Pairwise similarity  (39) — cosine, euclidean, nearest-neighbour
  Clustering            (8) — HDBSCAN sub-clusters, quality scores
  Information-theoretic (6) — character & word entropy
  Size                  (1) — forget_set_size
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
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN

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


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(v, default=0.0):
    return v if np.isfinite(v) else default


def _dist(values, pfx: str) -> dict:
    """13 descriptive statistics for a 1-D numeric array."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    _keys = ["mean", "std", "min", "max", "median",
             "q10", "q25", "q75", "q90", "range", "iqr", "skew", "kurtosis"]
    if n == 0:
        return {f"{pfx}_{k}": 0.0 for k in _keys}
    mn, mx = float(a.min()), float(a.max())
    q10, q25, med, q75, q90 = np.percentile(a, [10, 25, 50, 75, 90]).tolist()
    s = float(a.std())
    return {
        f"{pfx}_mean": float(a.mean()), f"{pfx}_std": s,
        f"{pfx}_min": mn, f"{pfx}_max": mx, f"{pfx}_median": med,
        f"{pfx}_q10": q10, f"{pfx}_q25": q25, f"{pfx}_q75": q75, f"{pfx}_q90": q90,
        f"{pfx}_range": mx - mn, f"{pfx}_iqr": q75 - q25,
        f"{pfx}_skew": _safe(float(sp_stats.skew(a))) if n >= 3 and s > 1e-9 else 0.0,
        f"{pfx}_kurtosis": _safe(float(sp_stats.kurtosis(a))) if n >= 4 and s > 1e-9 else 0.0,
    }


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _sents(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()] or [text]


# ── feature groups ───────────────────────────────────────────────────────────

def _lexical(texts: list[str], token_ids_list: list[list[int]]) -> dict:
    flat_ids = [t for ids in token_ids_list for t in ids]
    n_tok = len(flat_ids) or 1
    tc = Counter(flat_ids)
    uniq_tok = len(tc)
    hapax = sum(1 for v in tc.values() if v == 1)

    all_w: list[str] = []
    wlens: list[int] = []
    for t in texts:
        ws = _words(t)
        all_w.extend(ws)
        wlens.extend(len(w) for w in ws)
    n_w = len(all_w) or 1
    uniq_w = len(set(all_w))
    sw = sum(1 for w in all_w if w in _STOPWORDS)

    joined = "".join(texts)
    nc = len(joined) or 1
    punc = sum(1 for c in joined if c in string.punctuation)
    digs = sum(1 for c in joined if c.isdigit())
    ups = sum(1 for c in joined if c.isupper())

    swc = [len(_words(s)) for t in texts for s in _sents(t)]

    return {
        "unique_token_count": uniq_tok,
        "type_token_ratio": uniq_tok / n_tok,
        "hapax_ratio": hapax / max(uniq_tok, 1),
        "unique_word_count": uniq_w,
        "word_type_token_ratio": uniq_w / n_w,
        "avg_word_length": float(np.mean(wlens)) if wlens else 0.0,
        "word_length_std": float(np.std(wlens)) if wlens else 0.0,
        "long_word_ratio": sum(1 for l in wlens if l > 8) / n_w,
        "short_word_ratio": sum(1 for l in wlens if l <= 3) / n_w,
        "punctuation_ratio": punc / nc,
        "digit_ratio": digs / nc,
        "uppercase_ratio": ups / nc,
        "stopword_ratio": sw / n_w,
        "avg_sent_word_count": float(np.mean(swc)) if swc else 0.0,
        "sent_word_count_std": float(np.std(swc)) if swc else 0.0,
    }


def _emb_feats(embs: np.ndarray) -> dict:
    norms = np.linalg.norm(embs, axis=1)
    var = embs.var(axis=0)
    centroid = embs.mean(axis=0)
    n, d = embs.shape
    nc = max(min(n - 1, d), 1)
    pca = PCA(n_components=nc).fit(embs)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    p = ev + 1e-12

    f: dict = {}
    f.update(_dist(norms, "emb_norm"))
    f["emb_var_mean"] = float(var.mean())
    f["emb_var_max"] = float(var.max())
    f["emb_var_min"] = float(var.min())
    f["emb_var_std"] = float(var.std())
    f["emb_centroid_norm"] = float(np.linalg.norm(centroid))
    f["pca_var_top1"] = float(ev[0])
    f["pca_var_top3"] = float(ev[:3].sum()) if len(ev) >= 3 else float(ev.sum())
    f["pca_var_top5"] = float(ev[:5].sum()) if len(ev) >= 5 else float(ev.sum())
    f["pca_var_top10"] = float(ev[:10].sum()) if len(ev) >= 10 else float(ev.sum())
    f["pca_n90"] = min(int(np.searchsorted(cum, 0.90)) + 1, nc)
    f["pca_n95"] = min(int(np.searchsorted(cum, 0.95)) + 1, nc)
    f["pca_effective_rank"] = float(np.exp(-np.sum(p * np.log(p))))
    f["emb_isotropy"] = float(var.min() / max(float(var.max()), 1e-12))
    return f


def _sim_feats(embs: np.ndarray) -> dict:
    n = embs.shape[0]
    if n < 2:
        z: dict = {}
        for pfx in ("cos_sim", "eucl_dist", "nn_cos_dist"):
            z.update(_dist(np.zeros(1), pfx))
        return z

    sim = cosine_similarity(embs)
    triu = np.triu_indices(n, k=1)
    f: dict = {}
    f.update(_dist(sim[triu], "cos_sim"))

    dist = euclidean_distances(embs)
    f.update(_dist(dist[triu], "eucl_dist"))

    np.fill_diagonal(sim, -np.inf)
    nn_dist = 1.0 - sim.max(axis=1)
    f.update(_dist(nn_dist, "nn_cos_dist"))
    return f


def _cluster_feats(embs: np.ndarray) -> dict:
    n = embs.shape[0]
    nc_pca = max(min(20, n - 1, embs.shape[1]), 1)
    red = PCA(n_components=nc_pca).fit_transform(embs)
    labels = HDBSCAN(min_cluster_size=5).fit_predict(red)
    ul = set(labels) - {-1}
    nc = max(len(ul), 1)
    noise = labels == -1
    valid = ~noise

    f: dict = {"cluster_count": nc, "cluster_noise_frac": float(noise.sum() / n)}
    if nc >= 2 and valid.sum() >= nc:
        try:
            f["silhouette"] = float(silhouette_score(red[valid], labels[valid]))
            f["calinski_harabasz"] = float(calinski_harabasz_score(red[valid], labels[valid]))
            f["davies_bouldin"] = float(davies_bouldin_score(red[valid], labels[valid]))
        except Exception:
            f["silhouette"] = f["calinski_harabasz"] = f["davies_bouldin"] = 0.0
    else:
        f["silhouette"] = f["calinski_harabasz"] = f["davies_bouldin"] = 0.0

    if ul and valid.any():
        sizes = [int((labels == l).sum()) for l in ul]
        f["cluster_size_max"] = max(sizes)
        f["cluster_size_min"] = min(sizes)
        f["cluster_size_std"] = float(np.std(sizes)) if len(sizes) > 1 else 0.0
    else:
        f["cluster_size_max"] = f["cluster_size_min"] = n
        f["cluster_size_std"] = 0.0
    return f



def _info_feats(texts: list[str]) -> dict:
    ch, wh = [], []
    for t in texts:
        cc = Counter(t)
        nt = len(t) or 1
        p = np.array([v / nt for v in cc.values()])
        ch.append(float(-np.sum(p * np.log2(p + 1e-12))))
        ws = _words(t)
        if ws:
            wc = Counter(ws)
            nw = len(ws)
            p2 = np.array([v / nw for v in wc.values()])
            wh.append(float(-np.sum(p2 * np.log2(p2 + 1e-12))))
        else:
            wh.append(0.0)
    return {
        "char_entropy_mean": float(np.mean(ch)),
        "char_entropy_std": float(np.std(ch)),
        "char_entropy_min": float(np.min(ch)),
        "char_entropy_max": float(np.max(ch)),
        "word_entropy_mean": float(np.mean(wh)),
        "word_entropy_std": float(np.std(wh)),
    }


# ── per-triplet extraction ──────────────────────────────────────────────────

def extract_features(name, samples, embed_model, tokenizer):
    texts = [s["text"] for s in samples]

    tid = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

    embs = embed_model.encode(texts, show_progress_bar=False, batch_size=64)

    f: dict = {"split": name, "forget_set_size": len(samples)}

    # length distributions  (4 × 13 = 52)
    f.update(_dist([len(x) for x in tid], "text_tok"))
    f.update(_dist([len(t) for t in texts], "text_char"))
    f.update(_dist([len(_words(t)) for t in texts], "text_word"))
    f.update(_dist([len(_sents(t)) for t in texts], "text_sent"))

    # lexical (15)
    f.update(_lexical(texts, tid))
    # embedding geometry (26)
    f.update(_emb_feats(embs))
    # pairwise similarity (39)
    f.update(_sim_feats(embs))
    # clustering (8)
    f.update(_cluster_feats(embs))
    # info-theoretic (6)
    f.update(_info_feats(texts))

    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in f.items()}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract forget-set level features (~147 per triplet)"
    )
    ap.add_argument("--data_dir", type=str, default=str(WIKITEXT_DIR))
    ap.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--tokenizer", type=str, default="bert-base-cased")
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
    print(f"  Forget-Set Feature Extraction: {n} triplets")
    print(f"  Embed model : {args.embed_model}")
    print(f"  Tokenizer   : {args.tokenizer}")
    print("=" * 72)

    print("Loading models ...")
    embed_model = SentenceTransformer(args.embed_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("  Ready.\n")

    feats: list[dict] = []
    for i, td in enumerate(tdirs, 1):
        tp = td / "train.json"
        if not tp.exists():
            print(f"  [{i:3d}/{n}] {td.name} — skip")
            continue
        with open(tp) as fh:
            samples = json.load(fh)
        print(f"  [{i:3d}/{n}] {td.name} ({len(samples)}) ...", end=" ", flush=True)
        feat = extract_features(td.name, samples, embed_model, tokenizer)
        feats.append(feat)
        print(f"clusters={feat['cluster_count']}  emb_var={feat['emb_var_mean']:.4f}")

    df = pd.DataFrame(feats)
    n_feat = df.shape[1] - 1

    csv_path = out_dir / "forget_set_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  saved to {csv_path}  ({n_feat} features)")

    json_path = out_dir / "forget_set_features.json"
    with open(json_path, "w") as fh:
        json.dump(feats, fh, indent=2)
    print(f"JSON saved to {json_path}")

    print(f"\n── Columns ({n_feat} features) ──")
    for i, col in enumerate(df.columns, 1):
        if col != "split":
            print(f"  {i:3d}. {col}")


if __name__ == "__main__":
    main()
