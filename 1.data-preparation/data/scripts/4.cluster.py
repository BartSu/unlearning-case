"""
Run HDBSCAN clustering from saved reduced vectors.

Default input:
  data/wikitext_reduced/
    reduced_vectors.npy

Default outputs:
  data/wikitext_clusters_hdbscan/
    cluster_labels.npy
    cluster_distances.npy
    run_manifest.json

Optional subset artifacts when --subset_size is set:
  data/wikitext_clusters_hdbscan/
    reduced_vectors_subset.npy
    subset_indices.npy
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

from _hdbscan_pipeline_utils import (
    build_output_paths,
    clear_outputs_after_cluster,
    data_dir,
    default_cluster_output_dir,
    ensure_dir,
    load_json,
    now_utc_iso,
    run_hdbscan,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HDBSCAN from reduced_vectors.npy."
    )
    parser.add_argument(
        "--input_reduced",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_reduced/reduced_vectors.npy",
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size",
        type=int,
        default=200,
        help="Minimum cluster size passed to HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan_min_samples",
        type=int,
        default=5,
        help="Minimum samples parameter passed to HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan_metric",
        type=str,
        default="euclidean",
        help="Distance metric passed to sklearn.cluster.HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan_algorithm",
        choices=["auto", "ball_tree", "kd_tree", "brute"],
        default="auto",
        help="Nearest-neighbor search backend used by HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan_leaf_size",
        type=int,
        default=40,
        help="Leaf size passed to HDBSCAN when tree-based algorithms are used.",
    )
    parser.add_argument(
        "--hdbscan_n_jobs",
        type=int,
        default=-1,
        help="CPU workers for HDBSCAN neighbor search. None=1 worker, -1=all CPUs.",
    )
    parser.add_argument(
        "--hdbscan_cluster_selection_method",
        choices=["eom", "leaf"],
        default="eom",
        help="Cluster selection method passed to HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan_cluster_selection_epsilon",
        type=float,
        default=0.0,
        help="Cluster selection epsilon passed to HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan_allow_single_cluster",
        action="store_true",
        help="Allow HDBSCAN to return a single non-noise cluster.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_clusters_hdbscan",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=100000,
        help="Optional number of reduced vectors to cluster instead of the full dataset.",
    )
    parser.add_argument(
        "--subset_mode",
        choices=["head", "random"],
        default="random",
        help="How to choose rows when --subset_size is set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_input_reduced(explicit_input_reduced: str | None) -> str:
    if explicit_input_reduced:
        return os.path.abspath(explicit_input_reduced)
    return os.path.join(data_dir(), "wikitext_reduced", "reduced_vectors.npy")


def load_existing_manifest(manifest_path: str) -> Dict[str, object]:
    if not os.path.isfile(manifest_path):
        return {}
    return load_json(manifest_path)


def build_subset_output_paths(output_dir: str) -> Dict[str, str]:
    return {
        "subset_reduced_vectors_path": os.path.join(output_dir, "reduced_vectors_subset.npy"),
        "subset_indices_path": os.path.join(output_dir, "subset_indices.npy"),
    }


def clear_subset_outputs(output_dir: str, protected_paths: set[str] | None = None) -> None:
    protected = protected_paths or set()
    for path in build_subset_output_paths(output_dir).values():
        if path in protected:
            continue
        if os.path.exists(path):
            os.remove(path)


def inspect_reduced_input(input_reduced: str) -> tuple[int, int]:
    X = np.load(input_reduced, mmap_mode="r")
    n_texts = int(X.shape[0])
    reduced_dim = int(X.shape[1])
    del X
    return n_texts, reduced_dim


def select_subset_indices(
    n_texts: int,
    subset_size: int | None,
    subset_mode: str,
    seed: int,
) -> tuple[np.ndarray | None, Dict[str, object]]:
    subset_meta: Dict[str, object] = {
        "subset_applied": False,
        "subset_mode": None,
        "subset_requested_size": None,
        "subset_selected_size": int(n_texts),
        "original_n_texts": int(n_texts),
        "subset_indices_path": None,
        "subset_reduced_vectors_path": None,
        "subset_seed": None,
    }
    if subset_size is None:
        return None, subset_meta
    if subset_size <= 0:
        raise ValueError("--subset_size must be > 0")

    subset_meta["subset_mode"] = subset_mode
    subset_meta["subset_requested_size"] = int(subset_size)
    if subset_size >= n_texts:
        print(
            f"Requested subset_size={subset_size:,} covers all {n_texts:,} rows; "
            "clustering the full dataset.",
            flush=True,
        )
        return None, subset_meta

    selected_size = int(subset_size)
    subset_meta["subset_applied"] = True
    subset_meta["subset_selected_size"] = selected_size
    if subset_mode == "head":
        indices = np.arange(selected_size, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(n_texts, size=selected_size, replace=False).astype(
                np.int64, copy=False
            )
        )
        subset_meta["subset_seed"] = int(seed)
    return indices, subset_meta


def materialize_subset_input(
    input_reduced: str,
    output_dir: str,
    subset_indices: np.ndarray,
) -> Dict[str, str]:
    subset_paths = build_subset_output_paths(output_dir)
    X = np.load(input_reduced, mmap_mode="r")
    subset_vectors = np.asarray(X[subset_indices], dtype=np.float32)
    del X
    np.save(subset_paths["subset_reduced_vectors_path"], subset_vectors)
    np.save(subset_paths["subset_indices_path"], subset_indices)
    return subset_paths


def build_manifest(
    existing_manifest: Dict[str, object],
    source_input_reduced: str,
    clustering_input_reduced: str,
    output_paths: Dict[str, str],
    n_texts: int,
    original_n_texts: int,
    reduced_dim: int,
    cluster_meta: Dict[str, object],
) -> Dict[str, object]:
    now = now_utc_iso()
    manifest = dict(existing_manifest)
    manifest.setdefault("created_at_utc", now)
    manifest["updated_at_utc"] = now

    source = dict(manifest.get("source")) if isinstance(manifest.get("source"), dict) else {}
    source["reduced_vectors_path"] = source_input_reduced
    source["clustering_input_reduced_path"] = clustering_input_reduced
    manifest["source"] = source

    stats = dict(manifest.get("stats")) if isinstance(manifest.get("stats"), dict) else {}
    stats["n_texts"] = int(n_texts)
    stats["original_n_texts"] = int(original_n_texts)
    manifest["stats"] = stats

    reducer_meta = (
        dict(manifest.get("reducer")) if isinstance(manifest.get("reducer"), dict) else {}
    )
    reducer_meta.setdefault("reducer", None)
    reducer_meta["n_components"] = int(reduced_dim)
    reducer_meta["reduced_path"] = source_input_reduced
    manifest["reducer"] = reducer_meta
    manifest["clustering"] = cluster_meta

    manifest_outputs = dict(output_paths)
    manifest_outputs["reduced_vectors_path"] = source_input_reduced
    if clustering_input_reduced != source_input_reduced:
        manifest_outputs["subset_reduced_vectors_path"] = clustering_input_reduced
    subset_indices_path = cluster_meta.get("subset_indices_path")
    if isinstance(subset_indices_path, str):
        manifest_outputs["subset_indices_path"] = subset_indices_path
    manifest["outputs"] = manifest_outputs
    manifest["pipeline"] = {
        "step": "cluster",
        "status": "completed",
    }
    return manifest


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(default_cluster_output_dir(args.output_dir))
    output_paths = build_output_paths(output_dir)
    input_reduced = default_input_reduced(args.input_reduced)
    if not os.path.isfile(input_reduced):
        raise FileNotFoundError(f"input_reduced not found: {input_reduced}")

    n_texts, reduced_dim = inspect_reduced_input(input_reduced)
    if n_texts == 0:
        raise RuntimeError(
            "No reduced vectors available for clustering. Run 3.reduce_dimension.py first."
        )
    subset_indices, subset_meta = select_subset_indices(
        n_texts=n_texts,
        subset_size=args.subset_size,
        subset_mode=args.subset_mode,
        seed=args.seed,
    )
    original_n_texts = int(subset_meta["original_n_texts"])
    if subset_indices is not None:
        n_texts = int(subset_meta["subset_selected_size"])

    existing_manifest = load_existing_manifest(output_paths["manifest_json"])
    clear_outputs_after_cluster(output_paths)
    clear_subset_outputs(output_dir, protected_paths={input_reduced})

    clustering_input_reduced = input_reduced
    if subset_indices is not None:
        print(
            f"Preparing subset: mode={subset_meta['subset_mode']} "
            f"size={n_texts:,}/{original_n_texts:,}",
            flush=True,
        )
        subset_paths = materialize_subset_input(
            input_reduced=input_reduced,
            output_dir=output_dir,
            subset_indices=subset_indices,
        )
        clustering_input_reduced = subset_paths["subset_reduced_vectors_path"]
        subset_meta["subset_indices_path"] = subset_paths["subset_indices_path"]
        subset_meta["subset_reduced_vectors_path"] = subset_paths[
            "subset_reduced_vectors_path"
        ]

    print("Running HDBSCAN ...")
    labels, distances, cluster_meta = run_hdbscan(
        reduced_path=clustering_input_reduced,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric=args.hdbscan_metric,
        algorithm=args.hdbscan_algorithm,
        leaf_size=args.hdbscan_leaf_size,
        n_jobs=args.hdbscan_n_jobs,
        cluster_selection_method=args.hdbscan_cluster_selection_method,
        cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        allow_single_cluster=args.hdbscan_allow_single_cluster,
    )
    cluster_meta = dict(cluster_meta)
    cluster_meta["subset_applied"] = bool(subset_meta["subset_applied"])
    cluster_meta["subset_mode"] = subset_meta["subset_mode"]
    cluster_meta["subset_requested_size"] = subset_meta["subset_requested_size"]
    cluster_meta["subset_selected_size"] = int(subset_meta["subset_selected_size"])
    cluster_meta["original_n_texts"] = original_n_texts
    if subset_meta["subset_indices_path"] is not None:
        cluster_meta["subset_indices_path"] = subset_meta["subset_indices_path"]
    if subset_meta["subset_reduced_vectors_path"] is not None:
        cluster_meta["subset_reduced_vectors_path"] = subset_meta[
            "subset_reduced_vectors_path"
        ]
    if subset_meta["subset_seed"] is not None:
        cluster_meta["subset_seed"] = int(subset_meta["subset_seed"])
    np.save(output_paths["cluster_labels_path"], labels)
    np.save(output_paths["cluster_distances_path"], distances)

    manifest = build_manifest(
        existing_manifest=existing_manifest,
        source_input_reduced=input_reduced,
        clustering_input_reduced=clustering_input_reduced,
        output_paths=output_paths,
        n_texts=n_texts,
        original_n_texts=original_n_texts,
        reduced_dim=reduced_dim,
        cluster_meta=cluster_meta,
    )
    write_json(output_paths["manifest_json"], manifest)

    print("Done.")
    print(f"  labels:      {output_paths['cluster_labels_path']}")
    print(f"  distances:   {output_paths['cluster_distances_path']}")
    if cluster_meta["subset_applied"]:
        print(
            f"  subset:      {n_texts:,}/{original_n_texts:,} "
            f"({cluster_meta['subset_mode']})"
        )
        print(f"  subset_idx:  {cluster_meta['subset_indices_path']}")
    print(
        f"  clusters:    {cluster_meta['n_clusters_excluding_noise']} "
        f"(noise_ratio={cluster_meta['noise_ratio']:.4f})"
    )
    print(f"  manifest:    {output_paths['manifest_json']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
