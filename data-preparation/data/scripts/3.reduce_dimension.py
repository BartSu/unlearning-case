"""
Reduce saved embeddings for downstream clustering.

Default input:
  data/wikitext_embeddings/
    embeddings.npy

Default outputs:
  data/wikitext_reduced/
    reduced_vectors.npy
    run_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

from _hdbscan_pipeline_utils import (
    build_output_paths,
    clear_outputs_after_reduce,
    data_dir,
    ensure_dir,
    load_json,
    now_utc_iso,
    reduce_embeddings,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce saved embeddings and write reduced_vectors.npy."
    )
    parser.add_argument(
        "--input_embeddings",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_embeddings/embeddings.npy",
    )
    parser.add_argument("--reducer", choices=["pca", "umap"], default="pca")
    parser.add_argument("--n_components", type=int, default=100)
    parser.add_argument(
        "--pca_mode",
        choices=["auto", "incremental", "full"],
        default="auto",
        help="PCA mode. 'auto' picks incremental for larger arrays.",
    )
    parser.add_argument("--pca_batch_size", type=int, default=4096)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.0)
    parser.add_argument("--umap_metric", type=str, default="cosine")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_reduced",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_input_embeddings(explicit_input_embeddings: str | None) -> str:
    if explicit_input_embeddings:
        return os.path.abspath(explicit_input_embeddings)
    return os.path.join(data_dir(), "wikitext_embeddings", "embeddings.npy")


def default_output_dir(explicit_output_dir: str | None) -> str:
    if explicit_output_dir:
        return os.path.abspath(explicit_output_dir)
    return os.path.join(data_dir(), "wikitext_reduced")


def load_existing_manifest(manifest_path: str) -> Dict[str, object]:
    if not os.path.isfile(manifest_path):
        return {}
    return load_json(manifest_path)


def build_manifest(
    existing_manifest: Dict[str, object],
    input_embeddings: str,
    output_paths: Dict[str, str],
    n_texts: int,
    embedding_dim: int,
    reducer_meta: Dict[str, object],
) -> Dict[str, object]:
    now = now_utc_iso()
    manifest = dict(existing_manifest)
    manifest.setdefault("created_at_utc", now)
    manifest["updated_at_utc"] = now

    source = dict(manifest.get("source")) if isinstance(manifest.get("source"), dict) else {}
    source["embeddings_path"] = input_embeddings
    manifest["source"] = source

    stats = dict(manifest.get("stats")) if isinstance(manifest.get("stats"), dict) else {}
    stats["n_texts"] = int(n_texts)
    manifest["stats"] = stats

    embedding_meta = (
        dict(manifest.get("embedding")) if isinstance(manifest.get("embedding"), dict) else {}
    )
    embedding_meta.setdefault("embedding_model", None)
    embedding_meta.setdefault("embedding_batch_size", None)
    embedding_meta.setdefault("embedding_device", None)
    embedding_meta.setdefault("normalize_embeddings", None)
    embedding_meta["embedding_dim"] = int(embedding_dim)
    embedding_meta["embeddings_path"] = input_embeddings
    manifest["embedding"] = embedding_meta

    manifest["reducer"] = reducer_meta
    manifest["clustering"] = None

    manifest_outputs = dict(output_paths)
    manifest_outputs["embeddings_path"] = input_embeddings
    manifest["outputs"] = manifest_outputs
    manifest["pipeline"] = {
        "step": "reduce_dimension",
        "status": "completed",
    }
    return manifest


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(default_output_dir(args.output_dir))
    output_paths = build_output_paths(output_dir)
    input_embeddings = default_input_embeddings(args.input_embeddings)
    if not os.path.isfile(input_embeddings):
        raise FileNotFoundError(f"input_embeddings not found: {input_embeddings}")

    X = np.load(input_embeddings, mmap_mode="r")
    n_texts = int(X.shape[0])
    embedding_dim = int(X.shape[1])
    del X
    if n_texts == 0:
        raise RuntimeError(
            "No embeddings available for dimensionality reduction. Run 2.embed.py first."
        )

    existing_manifest = load_existing_manifest(output_paths["manifest_json"])
    clear_outputs_after_reduce(output_paths)

    print("Reducing dimensionality ...")
    reducer_meta = reduce_embeddings(
        embeddings_path=input_embeddings,
        reducer=args.reducer,
        n_components=args.n_components,
        seed=args.seed,
        reduced_path=output_paths["reduced_vectors_path"],
        pca_mode=args.pca_mode,
        pca_batch_size=args.pca_batch_size,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
    )

    manifest = build_manifest(
        existing_manifest=existing_manifest,
        input_embeddings=input_embeddings,
        output_paths=output_paths,
        n_texts=n_texts,
        embedding_dim=embedding_dim,
        reducer_meta=reducer_meta,
    )
    write_json(output_paths["manifest_json"], manifest)

    print("Done.")
    print(f"  reduced:  {output_paths['reduced_vectors_path']}")
    print(f"  manifest: {output_paths['manifest_json']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
