"""
Shared helpers for the split WikiText HDBSCAN pipeline scripts.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def default_filtered_dir() -> str:
    return os.path.join(data_dir(), "wikitext_filtered")


def default_cluster_output_dir(explicit_output_dir: Optional[str]) -> str:
    if explicit_output_dir:
        return os.path.abspath(explicit_output_dir)
    return os.path.join(data_dir(), "wikitext_clusters_hdbscan")


def default_filtered_texts_jsonl(explicit_path: Optional[str]) -> str:
    if explicit_path:
        return os.path.abspath(explicit_path)
    return os.path.join(default_filtered_dir(), "filtered_texts.jsonl")


def default_filtered_offsets_npy(explicit_path: Optional[str]) -> str:
    if explicit_path:
        return os.path.abspath(explicit_path)
    return os.path.join(default_filtered_dir(), "filtered_text_offsets.npy")


def default_filter_manifest_json(explicit_path: Optional[str]) -> str:
    if explicit_path:
        return os.path.abspath(explicit_path)
    return os.path.join(default_filtered_dir(), "filter_manifest.json")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_output_paths(output_dir: str) -> Dict[str, str]:
    return {
        "embeddings_path": os.path.join(output_dir, "embeddings.npy"),
        "reduced_vectors_path": os.path.join(output_dir, "reduced_vectors.npy"),
        "cluster_labels_path": os.path.join(output_dir, "cluster_labels.npy"),
        "cluster_distances_path": os.path.join(output_dir, "cluster_distances.npy"),
        "cluster_assignments_csv": os.path.join(output_dir, "cluster_assignments.csv"),
        "cluster_assignments_jsonl": os.path.join(output_dir, "cluster_assignments.jsonl"),
        "cluster_summary_json": os.path.join(output_dir, "cluster_summary.json"),
        "cluster_summary_csv": os.path.join(output_dir, "cluster_summary.csv"),
        "manifest_json": os.path.join(output_dir, "run_manifest.json"),
    }


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def count_offsets(offsets_path: str) -> int:
    offsets = np.load(offsets_path, mmap_mode="r")
    return int(offsets.shape[0])


def count_jsonl_records(jsonl_path: str) -> int:
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        return sum(1 for _ in fin)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def remove_if_exists(*paths: str) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def clear_outputs_after_embed(output_paths: Dict[str, str]) -> None:
    remove_if_exists(
        output_paths["embeddings_path"],
        output_paths["reduced_vectors_path"],
        output_paths["cluster_labels_path"],
        output_paths["cluster_distances_path"],
        output_paths["cluster_assignments_csv"],
        output_paths["cluster_assignments_jsonl"],
        output_paths["cluster_summary_json"],
        output_paths["cluster_summary_csv"],
    )


def clear_outputs_after_reduce(output_paths: Dict[str, str]) -> None:
    remove_if_exists(
        output_paths["reduced_vectors_path"],
        output_paths["cluster_labels_path"],
        output_paths["cluster_distances_path"],
        output_paths["cluster_assignments_csv"],
        output_paths["cluster_assignments_jsonl"],
        output_paths["cluster_summary_json"],
        output_paths["cluster_summary_csv"],
    )


def clear_outputs_after_cluster(output_paths: Dict[str, str]) -> None:
    remove_if_exists(
        output_paths["cluster_labels_path"],
        output_paths["cluster_distances_path"],
        output_paths["cluster_assignments_csv"],
        output_paths["cluster_assignments_jsonl"],
        output_paths["cluster_summary_json"],
        output_paths["cluster_summary_csv"],
    )


def clear_outputs_after_summarize(output_paths: Dict[str, str]) -> None:
    remove_if_exists(
        output_paths["cluster_summary_json"],
        output_paths["cluster_summary_csv"],
        output_paths["cluster_assignments_csv"],
        output_paths["cluster_assignments_jsonl"],
    )


def iter_text_batches(texts_path: str, batch_size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    with open(texts_path, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            batch.append(rec["text"])
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def create_embeddings(
    texts_path: str,
    n_texts: int,
    model_name: str,
    batch_size: int,
    device: str,
    normalize_embeddings: bool,
    embeddings_path: str,
) -> Dict[str, object]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'sentence-transformers'. Install: pip install sentence-transformers"
        ) from exc

    model_kwargs = {}
    if device != "auto":
        model_kwargs["device"] = device
    model = SentenceTransformer(model_name, **model_kwargs)

    emb_dim = model.get_sentence_embedding_dimension()
    if emb_dim is None:
        for first_batch in iter_text_batches(texts_path, 1):
            probe = model.encode(
                first_batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings,
            )
            emb_dim = int(probe.shape[1])
            break
    if emb_dim is None:
        raise RuntimeError("Failed to infer embedding dimension.")

    mmap = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_texts, int(emb_dim)),
    )

    total_batches = int(math.ceil(n_texts / batch_size))
    cursor = 0
    for i, batch in enumerate(iter_text_batches(texts_path, batch_size), start=1):
        arr = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
        )
        arr = np.asarray(arr, dtype=np.float32)
        n_rows = arr.shape[0]
        mmap[cursor : cursor + n_rows] = arr
        cursor += n_rows
        if i % 25 == 0 or i == total_batches:
            print(f"[embed] batch {i}/{total_batches} ({cursor}/{n_texts} texts)")

    mmap.flush()
    del mmap
    if cursor != n_texts:
        raise RuntimeError(f"Embedding count mismatch: expected {n_texts}, got {cursor}")

    return {
        "embedding_model": model_name,
        "embedding_dim": int(emb_dim),
        "embedding_device": str(getattr(model, "device", device)),
        "embedding_batch_size": batch_size,
        "normalize_embeddings": bool(normalize_embeddings),
        "embeddings_path": embeddings_path,
    }


def reduce_embeddings(
    embeddings_path: str,
    reducer: str,
    n_components: int,
    seed: int,
    reduced_path: str,
    pca_mode: str,
    pca_batch_size: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
) -> Dict[str, object]:
    from sklearn.decomposition import IncrementalPCA, PCA

    X = np.load(embeddings_path, mmap_mode="r")
    n_samples, n_features = X.shape
    if n_components <= 0:
        raise ValueError("--n_components must be > 0")
    if n_components >= n_features:
        raise ValueError(
            f"--n_components ({n_components}) must be < embedding dim ({n_features})"
        )

    if reducer == "pca":
        if pca_mode == "auto":
            mode = "incremental" if n_samples > (2 * pca_batch_size) else "full"
        else:
            mode = pca_mode

        if mode == "incremental":
            ipca = IncrementalPCA(n_components=n_components, batch_size=pca_batch_size)
            for start in range(0, n_samples, pca_batch_size):
                end = min(start + pca_batch_size, n_samples)
                ipca.partial_fit(X[start:end])

            reduced_mmap = np.lib.format.open_memmap(
                reduced_path,
                mode="w+",
                dtype=np.float32,
                shape=(n_samples, n_components),
            )
            for start in range(0, n_samples, pca_batch_size):
                end = min(start + pca_batch_size, n_samples)
                reduced_mmap[start:end] = ipca.transform(X[start:end]).astype(np.float32)
            reduced_mmap.flush()
            del reduced_mmap

            return {
                "reducer": "pca",
                "pca_mode": "incremental",
                "n_components": n_components,
                "explained_variance_ratio_sum": float(np.sum(ipca.explained_variance_ratio_)),
                "reduced_path": reduced_path,
            }

        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        reduced = pca.fit_transform(X).astype(np.float32)
        np.save(reduced_path, reduced)
        return {
            "reducer": "pca",
            "pca_mode": "full",
            "n_components": n_components,
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "reduced_path": reduced_path,
        }

    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'umap-learn'. Install: pip install umap-learn") from exc

    reducer_obj = umap.UMAP(
        n_components=n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=seed,
        low_memory=True,
        verbose=True,
    )
    reduced = reducer_obj.fit_transform(X).astype(np.float32)
    np.save(reduced_path, reduced)
    return {
        "reducer": "umap",
        "n_components": n_components,
        "umap_n_neighbors": umap_n_neighbors,
        "umap_min_dist": umap_min_dist,
        "umap_metric": umap_metric,
        "reduced_path": reduced_path,
    }


def summarize_cluster_labels(labels: np.ndarray) -> Dict[str, object]:
    n_samples = int(labels.shape[0])
    unique_labels = sorted({int(label) for label in labels.tolist()})
    non_noise_labels = [label for label in unique_labels if label != -1]
    cluster_sizes = sorted(
        [int(np.sum(labels == label)) for label in non_noise_labels],
        reverse=True,
    )
    noise_count = int(np.sum(labels == -1))
    return {
        "n_clusters_excluding_noise": len(non_noise_labels),
        "noise_count": noise_count,
        "noise_ratio": float(noise_count / n_samples) if n_samples else 0.0,
        "largest_cluster_size": cluster_sizes[0] if cluster_sizes else 0,
        "smallest_cluster_size": cluster_sizes[-1] if cluster_sizes else 0,
        "median_cluster_size": (
            int(np.median(np.asarray(cluster_sizes, dtype=np.int64))) if cluster_sizes else 0
        ),
    }


def compute_posthoc_centroid_distances(
    X: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 200000,
    log_every_clusters: int = 25,
) -> np.ndarray:
    distances = np.full(labels.shape[0], np.nan, dtype=np.float32)
    unique_labels = sorted({int(label) for label in labels.tolist() if int(label) != -1})
    n_clusters = len(unique_labels)

    if n_clusters == 0:
        print("[cluster] no non-noise clusters found; skipping centroid distances.", flush=True)
        return distances

    for cluster_idx, cluster_id in enumerate(unique_labels, start=1):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        if cluster_indices.size == 0:
            continue

        centroid = np.asarray(np.mean(X[cluster_indices], axis=0), dtype=np.float32)
        for start in range(0, cluster_indices.size, batch_size):
            batch_idx = cluster_indices[start : start + batch_size]
            distances[batch_idx] = np.linalg.norm(X[batch_idx] - centroid, axis=1).astype(
                np.float32
            )

        if log_every_clusters > 0 and (
            cluster_idx % log_every_clusters == 0 or cluster_idx == n_clusters
        ):
            print(
                f"[cluster/distances] processed {cluster_idx}/{n_clusters} clusters",
                flush=True,
            )

    return distances


def _current_process_rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fin:
            for line in fin:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) / 1024.0
                    break
    except OSError:
        return None
    return None


def _start_heartbeat(label: str, interval_seconds: float = 30.0) -> Tuple[threading.Event, threading.Thread, float]:
    stop_event = threading.Event()
    started_at = time.perf_counter()

    def worker() -> None:
        while not stop_event.wait(interval_seconds):
            elapsed_minutes = (time.perf_counter() - started_at) / 60.0
            rss_mb = _current_process_rss_mb()
            rss_text = f" rss={rss_mb:.0f}MB" if rss_mb is not None else ""
            print(
                f"[{label}] still running ... elapsed={elapsed_minutes:.1f} min{rss_text}",
                flush=True,
            )

    thread = threading.Thread(target=worker, name=f"{label}-heartbeat", daemon=True)
    thread.start()
    return stop_event, thread, started_at


def _stop_heartbeat(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    thread.join(timeout=0.1)


def run_hdbscan(
    reduced_path: str,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    algorithm: str,
    leaf_size: int,
    n_jobs: Optional[int],
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    allow_single_cluster: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "Missing HDBSCAN support in scikit-learn. Install a recent scikit-learn release."
        ) from exc

    X = np.load(reduced_path, mmap_mode="r")
    n_samples, n_features = X.shape

    if min_cluster_size <= 1:
        raise ValueError("--hdbscan_min_cluster_size must be > 1")
    if min_samples <= 0:
        raise ValueError("--hdbscan_min_samples must be > 0")
    if cluster_selection_epsilon < 0.0:
        raise ValueError("--hdbscan_cluster_selection_epsilon must be >= 0")
    if n_samples == 0:
        raise RuntimeError("No rows available for clustering.")

    if n_samples > 100000:
        print(
            "Warning: HDBSCAN can still be slow and memory-heavy for large corpora. "
            "Consider tuning --hdbscan_min_cluster_size or --hdbscan_min_samples."
        )

    estimator = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=n_jobs,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=allow_single_cluster,
    )

    effective_workers = os.cpu_count() if n_jobs == -1 else (1 if n_jobs is None else n_jobs)
    print(
        f"[cluster] fit_predict start: samples={n_samples:,} dims={n_features:,} "
        f"metric={metric} algorithm={algorithm} workers~={effective_workers}",
        flush=True,
    )
    fit_stop_event, fit_thread, fit_started_at = _start_heartbeat("cluster/fit")
    try:
        labels = estimator.fit_predict(X).astype(np.int32)
    finally:
        _stop_heartbeat(fit_stop_event, fit_thread)

    fit_elapsed_seconds = time.perf_counter() - fit_started_at
    label_summary = summarize_cluster_labels(labels)
    print(
        f"[cluster] fit_predict done in {fit_elapsed_seconds / 60.0:.1f} min; "
        f"clusters={label_summary['n_clusters_excluding_noise']} "
        f"noise_ratio={label_summary['noise_ratio']:.4f}",
        flush=True,
    )

    print("[cluster] computing post-hoc centroid distances ...", flush=True)
    distance_started_at = time.perf_counter()
    distances = compute_posthoc_centroid_distances(
        X=X,
        labels=labels,
        log_every_clusters=25,
    )
    distance_elapsed_seconds = time.perf_counter() - distance_started_at
    print(
        f"[cluster] centroid distances done in {distance_elapsed_seconds / 60.0:.1f} min",
        flush=True,
    )

    cluster_meta = {
        "algorithm": "hdbscan",
        "hdbscan_min_cluster_size": min_cluster_size,
        "hdbscan_min_samples": min_samples,
        "hdbscan_metric": metric,
        "hdbscan_algorithm": algorithm,
        "hdbscan_leaf_size": leaf_size,
        "hdbscan_n_jobs": n_jobs,
        "hdbscan_cluster_selection_method": cluster_selection_method,
        "hdbscan_cluster_selection_epsilon": cluster_selection_epsilon,
        "hdbscan_allow_single_cluster": allow_single_cluster,
        **label_summary,
    }
    return labels, distances, cluster_meta


def read_text_at_offset(fp, offset: int) -> str:
    fp.seek(int(offset))
    line = fp.readline()
    if not line:
        return ""
    return json.loads(line)["text"]


def sample_indices(indices: Sequence[int], max_docs: int, seed: int) -> List[int]:
    if len(indices) <= max_docs:
        return list(indices)
    rng = random.Random(seed)
    sampled = rng.sample(list(indices), max_docs)
    sampled.sort()
    return sampled


def extract_top_keywords(
    docs: Sequence[str],
    top_k: int,
    min_df: int,
    max_features: int,
) -> List[str]:
    if not docs:
        return []

    from sklearn.feature_extraction.text import CountVectorizer

    token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b"
    effective_min_df = min_df
    while effective_min_df >= 1:
        try:
            vectorizer = CountVectorizer(
                stop_words="english",
                lowercase=True,
                token_pattern=token_pattern,
                min_df=effective_min_df,
                max_features=max_features,
            )
            mat = vectorizer.fit_transform(docs)
            terms = vectorizer.get_feature_names_out()
            freqs = np.asarray(mat.sum(axis=0)).ravel()
            if freqs.size == 0:
                return []
            top_ids = np.argsort(freqs)[::-1][:top_k]
            return [str(terms[i]) for i in top_ids if freqs[i] > 0]
        except ValueError:
            effective_min_df -= 1
    return []


def infer_domain_name(cluster_id: int, keywords: Sequence[str]) -> str:
    if cluster_id == -1:
        return "noise"
    if not keywords:
        return f"cluster_{cluster_id}"
    return "_".join(keywords[:3])


def build_cluster_summary(
    labels: np.ndarray,
    texts_path: str,
    offsets_path: str,
    top_k_keywords: int,
    keyword_min_df: int,
    keyword_max_features: int,
    keyword_max_docs_per_cluster: int,
    seed: int,
) -> Tuple[Dict[int, str], Dict[str, object]]:
    offsets = np.load(offsets_path, mmap_mode="r")
    total = len(labels)
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        cluster_to_indices[int(label)].append(idx)

    cluster_labels: Dict[int, str] = {}
    cluster_top_terms: Dict[int, List[str]] = {}
    cluster_rows: List[Dict[str, object]] = []

    with open(texts_path, "r", encoding="utf-8") as fp:
        for cluster_id in sorted(cluster_to_indices):
            indices = cluster_to_indices[cluster_id]
            size = len(indices)

            if cluster_id == -1:
                keywords: List[str] = []
                domain = "noise"
            else:
                sampled = sample_indices(
                    indices=indices,
                    max_docs=keyword_max_docs_per_cluster,
                    seed=seed + (cluster_id * 7919),
                )
                docs = [read_text_at_offset(fp, int(offsets[i])) for i in sampled]
                keywords = extract_top_keywords(
                    docs=docs,
                    top_k=top_k_keywords,
                    min_df=keyword_min_df,
                    max_features=keyword_max_features,
                )
                domain = infer_domain_name(cluster_id, keywords)

            cluster_labels[cluster_id] = domain
            cluster_top_terms[cluster_id] = keywords
            cluster_rows.append(
                {
                    "cluster_label": cluster_id,
                    "domain": domain,
                    "size": size,
                    "ratio": float(size / total),
                    "is_noise": bool(cluster_id == -1),
                    "keywords": keywords,
                }
            )

    summary = {
        "n_texts": total,
        "n_clusters_excluding_noise": len([cluster_id for cluster_id in cluster_to_indices if cluster_id != -1]),
        "noise_count": len(cluster_to_indices.get(-1, [])),
        "noise_ratio": float(len(cluster_to_indices.get(-1, [])) / total) if total else 0.0,
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "cluster_top_terms": {str(k): v for k, v in cluster_top_terms.items()},
        "clusters": cluster_rows,
    }
    return cluster_labels, summary


def write_cluster_summary_csv(summary: Dict[str, object], csv_path: str) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as fout:
        fieldnames = ["cluster_label", "domain", "size", "ratio", "is_noise", "keywords"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["clusters"]:
            writer.writerow(
                {
                    "cluster_label": row["cluster_label"],
                    "domain": row["domain"],
                    "size": row["size"],
                    "ratio": f"{float(row['ratio']):.8f}",
                    "is_noise": row["is_noise"],
                    "keywords": "|".join(row["keywords"]),
                }
            )


def load_cluster_labels_from_summary(summary_json_path: str) -> Dict[int, str]:
    summary = load_json(summary_json_path)
    raw_cluster_labels = summary.get("cluster_labels")
    if not isinstance(raw_cluster_labels, dict):
        raise RuntimeError(f"Missing 'cluster_labels' mapping in {summary_json_path}")

    cluster_labels: Dict[int, str] = {}
    for key, value in raw_cluster_labels.items():
        cluster_labels[int(key)] = str(value)
    return cluster_labels


def iter_texts(texts_path: str) -> Iterator[str]:
    with open(texts_path, "r", encoding="utf-8") as fin:
        for line in fin:
            yield json.loads(line)["text"]


def write_assignments(
    labels: np.ndarray,
    distances: np.ndarray,
    cluster_labels: Dict[int, str],
    csv_path: str,
    jsonl_path: str,
    include_text: bool,
    texts_path: str,
) -> None:
    fieldnames = ["text_id", "cluster_label", "domain", "is_noise", "distance_to_centroid"]
    if include_text:
        fieldnames.append("text")

    text_iter: Optional[Iterator[str]] = iter_texts(texts_path) if include_text else None

    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv, open(
        jsonl_path, "w", encoding="utf-8"
    ) as f_jsonl:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for idx, label in enumerate(labels.tolist()):
            domain = cluster_labels.get(int(label), "noise" if int(label) == -1 else f"cluster_{label}")
            distance = float(distances[idx]) if np.isfinite(distances[idx]) else None
            record = {
                "text_id": idx,
                "cluster_label": int(label),
                "domain": domain,
                "is_noise": int(label) == -1,
                "distance_to_centroid": "" if distance is None else f"{distance:.6f}",
            }
            json_record = {
                "text_id": idx,
                "cluster_label": int(label),
                "domain": domain,
                "is_noise": int(label) == -1,
                "distance_to_centroid": distance,
            }

            if include_text:
                assert text_iter is not None
                text = next(text_iter)
                record["text"] = text
                json_record["text"] = text

            writer.writerow(record)
            f_jsonl.write(json.dumps(json_record, ensure_ascii=False) + "\n")
