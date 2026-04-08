"""
Build cluster summary files from saved HDBSCAN labels.

Default inputs:
  data/wikitext_clusters_hdbscan/
    cluster_labels.npy
  data/wikitext_filtered/
    filtered_texts.jsonl
    filtered_text_offsets.npy

Default outputs:
  data/wikitext_clusters_hdbscan_summary/
    cluster_summary.json
    cluster_summary.csv
    run_manifest.json

Optional subset artifact when cluster labels were computed on a subset:
  data/wikitext_clusters_hdbscan_summary/
    filtered_text_offsets_subset.npy
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

import numpy as np

from _hdbscan_pipeline_utils import (
    build_cluster_summary,
    build_output_paths,
    clear_outputs_after_summarize,
    count_offsets,
    default_cluster_output_dir,
    default_filtered_offsets_npy,
    default_filtered_texts_jsonl,
    ensure_dir,
    load_json,
    now_utc_iso,
    set_seed,
    write_cluster_summary_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cluster_summary.json/csv from cluster_labels.npy."
    )
    parser.add_argument(
        "--input_labels",
        type=str,
        default=None,
        help="Default: <output_dir>/cluster_labels.npy",
    )
    parser.add_argument(
        "--texts_jsonl",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_filtered/filtered_texts.jsonl",
    )
    parser.add_argument(
        "--offsets_npy",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_filtered/filtered_text_offsets.npy",
    )
    parser.add_argument("--top_k_keywords", type=int, default=15)
    parser.add_argument("--keyword_min_df", type=int, default=2)
    parser.add_argument("--keyword_max_features", type=int, default=50000)
    parser.add_argument(
        "--keyword_max_docs_per_cluster",
        type=int,
        default=5000,
        help="Sample at most this many docs per cluster for keyword extraction.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Cluster output dir. Default: <data_dir>/wikitext_clusters_hdbscan",
    )
    parser.add_argument(
        "--summary_output_dir",
        type=str,
        default=None,
        help="Summary output dir. Default: <output_dir>_summary",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_existing_manifest(manifest_path: str) -> Dict[str, object]:
    if not os.path.isfile(manifest_path):
        return {}
    return load_json(manifest_path)


def default_input_labels(explicit_input_labels: Optional[str], output_paths: Dict[str, str]) -> str:
    if explicit_input_labels:
        return os.path.abspath(explicit_input_labels)
    return output_paths["cluster_labels_path"]


def default_summary_output_dir(
    explicit_summary_output_dir: Optional[str], cluster_output_dir: str
) -> str:
    if explicit_summary_output_dir:
        return os.path.abspath(explicit_summary_output_dir)
    return f"{os.path.abspath(cluster_output_dir)}_summary"


def default_subset_offsets_npy(summary_output_dir: str) -> str:
    return os.path.join(summary_output_dir, "filtered_text_offsets_subset.npy")


def discover_subset_indices_path(
    existing_manifest: Dict[str, object], cluster_output_dir: str
) -> Optional[str]:
    candidates = []
    outputs = existing_manifest.get("outputs")
    if isinstance(outputs, dict):
        candidates.append(outputs.get("subset_indices_path"))
    clustering = existing_manifest.get("clustering")
    if isinstance(clustering, dict):
        candidates.append(clustering.get("subset_indices_path"))
    candidates.append(os.path.join(cluster_output_dir, "subset_indices.npy"))

    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        path = os.path.abspath(candidate)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


def resolve_summary_offsets(
    labels: np.ndarray,
    offsets_npy: str,
    existing_manifest: Dict[str, object],
    cluster_output_dir: str,
    summary_output_dir: str,
) -> tuple[str, Dict[str, object]]:
    n_texts = int(labels.shape[0])
    n_offsets = count_offsets(offsets_npy)
    offsets_meta: Dict[str, object] = {
        "base_offsets_npy": offsets_npy,
        "summary_offsets_npy": offsets_npy,
        "subset_indices_path": None,
        "subset_offsets_npy": None,
        "original_n_offsets": int(n_offsets),
    }
    if n_offsets == n_texts:
        return offsets_npy, offsets_meta

    subset_indices_path = discover_subset_indices_path(existing_manifest, cluster_output_dir)
    if subset_indices_path is None:
        raise RuntimeError(
            f"Label/text count mismatch: cluster_labels={n_texts}, filtered_offsets={n_offsets}"
        )

    subset_indices = np.load(subset_indices_path, mmap_mode="r")
    if subset_indices.ndim != 1:
        raise RuntimeError(
            f"Expected 1D subset indices in {subset_indices_path}, got shape={subset_indices.shape}"
        )
    if int(subset_indices.shape[0]) != n_texts:
        raise RuntimeError(
            "Label/subset count mismatch: "
            f"cluster_labels={n_texts}, subset_indices={int(subset_indices.shape[0])}"
        )

    if subset_indices.size > 0:
        min_index = int(np.min(subset_indices))
        max_index = int(np.max(subset_indices))
        if min_index < 0 or max_index >= n_offsets:
            raise RuntimeError(
                "subset_indices contain out-of-range values for the provided filtered offsets: "
                f"min={min_index}, max={max_index}, filtered_offsets={n_offsets}"
            )

    subset_offsets_npy = default_subset_offsets_npy(summary_output_dir)
    full_offsets = np.load(offsets_npy, mmap_mode="r")
    subset_offsets = np.asarray(full_offsets[subset_indices], dtype=full_offsets.dtype)
    del full_offsets
    np.save(subset_offsets_npy, subset_offsets)

    print(
        f"Detected subset clustering outputs; aligned offsets: {n_texts:,}/{n_offsets:,}",
        flush=True,
    )
    print(f"  subset_indices: {subset_indices_path}", flush=True)
    print(f"  subset_offsets: {subset_offsets_npy}", flush=True)

    offsets_meta["summary_offsets_npy"] = subset_offsets_npy
    offsets_meta["subset_indices_path"] = subset_indices_path
    offsets_meta["subset_offsets_npy"] = subset_offsets_npy
    return subset_offsets_npy, offsets_meta


def build_manifest(
    existing_manifest: Dict[str, object],
    input_labels: str,
    texts_jsonl: str,
    base_offsets_npy: str,
    summary_offsets_npy: str,
    subset_indices_path: Optional[str],
    summary_output_paths: Dict[str, str],
    n_texts: int,
    summary: Dict[str, object],
    top_k_keywords: int,
    keyword_min_df: int,
    keyword_max_features: int,
    keyword_max_docs_per_cluster: int,
) -> Dict[str, object]:
    now = now_utc_iso()
    manifest = dict(existing_manifest)
    manifest.setdefault("created_at_utc", now)
    manifest["updated_at_utc"] = now

    source = dict(manifest.get("source")) if isinstance(manifest.get("source"), dict) else {}
    source["cluster_labels_path"] = input_labels
    source["filtered_texts_jsonl"] = texts_jsonl
    source["filtered_offsets_npy"] = base_offsets_npy
    source["summary_offsets_npy"] = summary_offsets_npy
    if subset_indices_path is not None:
        source["subset_indices_path"] = subset_indices_path
    else:
        source.pop("subset_indices_path", None)
    manifest["source"] = source

    stats = dict(manifest.get("stats")) if isinstance(manifest.get("stats"), dict) else {}
    stats["n_texts"] = int(n_texts)
    manifest["stats"] = stats

    keywording = {
        "top_k_keywords": top_k_keywords,
        "keyword_min_df": keyword_min_df,
        "keyword_max_features": keyword_max_features,
        "keyword_max_docs_per_cluster": keyword_max_docs_per_cluster,
    }
    manifest["keywording"] = keywording

    clustering_meta = (
        dict(manifest.get("clustering")) if isinstance(manifest.get("clustering"), dict) else {}
    )
    clustering_meta["n_clusters_excluding_noise"] = int(summary["n_clusters_excluding_noise"])
    clustering_meta["noise_count"] = int(summary["noise_count"])
    clustering_meta["noise_ratio"] = float(summary["noise_ratio"])
    manifest["clustering"] = clustering_meta

    manifest_outputs = (
        dict(manifest.get("outputs")) if isinstance(manifest.get("outputs"), dict) else {}
    )
    manifest_outputs["cluster_summary_json"] = summary_output_paths["cluster_summary_json"]
    manifest_outputs["cluster_summary_csv"] = summary_output_paths["cluster_summary_csv"]
    manifest_outputs["manifest_json"] = summary_output_paths["manifest_json"]
    manifest_outputs["cluster_labels_path"] = input_labels
    if summary_offsets_npy != base_offsets_npy:
        manifest_outputs["subset_filtered_offsets_npy"] = summary_offsets_npy
    else:
        manifest_outputs.pop("subset_filtered_offsets_npy", None)
    if subset_indices_path is not None:
        manifest_outputs["subset_indices_path"] = subset_indices_path
    manifest["outputs"] = manifest_outputs
    manifest["pipeline"] = {
        "step": "summarize",
        "status": "completed",
    }
    return manifest


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cluster_output_dir = ensure_dir(default_cluster_output_dir(args.output_dir))
    cluster_output_paths = build_output_paths(cluster_output_dir)
    summary_output_dir = ensure_dir(
        default_summary_output_dir(args.summary_output_dir, cluster_output_dir)
    )
    summary_output_paths = build_output_paths(summary_output_dir)
    input_labels = default_input_labels(args.input_labels, cluster_output_paths)
    texts_jsonl = default_filtered_texts_jsonl(args.texts_jsonl)
    offsets_npy = default_filtered_offsets_npy(args.offsets_npy)

    if not os.path.isfile(input_labels):
        raise FileNotFoundError(f"input_labels not found: {input_labels}")
    if not os.path.isfile(texts_jsonl):
        raise FileNotFoundError(f"texts_jsonl not found: {texts_jsonl}")
    if not os.path.isfile(offsets_npy):
        raise FileNotFoundError(f"offsets_npy not found: {offsets_npy}")

    cluster_manifest = load_existing_manifest(cluster_output_paths["manifest_json"])
    existing_summary_manifest = load_existing_manifest(summary_output_paths["manifest_json"])
    existing_manifest = dict(cluster_manifest) if cluster_manifest else dict(existing_summary_manifest)
    if "created_at_utc" in existing_summary_manifest:
        existing_manifest["created_at_utc"] = existing_summary_manifest["created_at_utc"]
    labels = np.load(input_labels, mmap_mode="r")
    n_texts = int(labels.shape[0])
    if n_texts == 0:
        raise RuntimeError("No cluster labels available for summarization. Run 4.cluster.py first.")
    summary_offsets_npy, offsets_meta = resolve_summary_offsets(
        labels=labels,
        offsets_npy=offsets_npy,
        existing_manifest=cluster_manifest if cluster_manifest else existing_manifest,
        cluster_output_dir=cluster_output_dir,
        summary_output_dir=summary_output_dir,
    )

    clear_outputs_after_summarize(summary_output_paths)

    print("Building cluster summary ...")
    _, summary = build_cluster_summary(
        labels=labels,
        texts_path=texts_jsonl,
        offsets_path=summary_offsets_npy,
        top_k_keywords=args.top_k_keywords,
        keyword_min_df=args.keyword_min_df,
        keyword_max_features=args.keyword_max_features,
        keyword_max_docs_per_cluster=args.keyword_max_docs_per_cluster,
        seed=args.seed,
    )
    write_json(summary_output_paths["cluster_summary_json"], summary)
    write_cluster_summary_csv(
        summary=summary, csv_path=summary_output_paths["cluster_summary_csv"]
    )

    manifest = build_manifest(
        existing_manifest=existing_manifest,
        input_labels=input_labels,
        texts_jsonl=texts_jsonl,
        base_offsets_npy=offsets_npy,
        summary_offsets_npy=summary_offsets_npy,
        subset_indices_path=(
            str(offsets_meta["subset_indices_path"])
            if offsets_meta["subset_indices_path"] is not None
            else None
        ),
        summary_output_paths=summary_output_paths,
        n_texts=n_texts,
        summary=summary,
        top_k_keywords=args.top_k_keywords,
        keyword_min_df=args.keyword_min_df,
        keyword_max_features=args.keyword_max_features,
        keyword_max_docs_per_cluster=args.keyword_max_docs_per_cluster,
    )
    write_json(summary_output_paths["manifest_json"], manifest)

    print("Done.")
    print(f"  summary_json: {summary_output_paths['cluster_summary_json']}")
    print(f"  summary_csv:  {summary_output_paths['cluster_summary_csv']}")
    if summary_offsets_npy != offsets_npy:
        print(f"  subset_offsets:{summary_offsets_npy}")
    print(
        f"  clusters:     {summary['n_clusters_excluding_noise']} "
        f"(noise_ratio={summary['noise_ratio']:.4f})"
    )
    print(f"  manifest:     {summary_output_paths['manifest_json']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
