"""
Export per-text cluster assignments from saved HDBSCAN outputs.

Default inputs:
  data/wikitext_clusters_hdbscan/
    cluster_labels.npy
    cluster_distances.npy
  data/wikitext_clusters_hdbscan_summary/
    cluster_summary.json
  data/wikitext_filtered/
    filtered_texts.jsonl

Default outputs:
  data/wikitext_clusters_hdbscan_export/
    cluster_assignments.csv
    cluster_assignments.jsonl
    run_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

import numpy as np

from _hdbscan_pipeline_utils import (
    build_output_paths,
    count_jsonl_records,
    default_cluster_output_dir,
    default_filtered_texts_jsonl,
    ensure_dir,
    load_cluster_labels_from_summary,
    load_json,
    now_utc_iso,
    write_assignments,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cluster_assignments.csv/jsonl from saved clustering outputs."
    )
    parser.add_argument(
        "--input_labels",
        type=str,
        default=None,
        help="Default: <output_dir>/cluster_labels.npy",
    )
    parser.add_argument(
        "--input_distances",
        type=str,
        default=None,
        help="Default: <output_dir>/cluster_distances.npy",
    )
    parser.add_argument(
        "--input_summary",
        type=str,
        default=None,
        help="Default: <summary_output_dir>/cluster_summary.json",
    )
    parser.add_argument(
        "--texts_jsonl",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_filtered/filtered_texts.jsonl",
    )
    parser.add_argument(
        "--include_text_in_assignments",
        action="store_true",
        help="Include raw text in assignment exports (larger files).",
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
        help="Summary dir from 5.summarize.py. Default: <output_dir>_summary",
    )
    parser.add_argument(
        "--export_output_dir",
        type=str,
        default=None,
        help="Export output dir. Default: <output_dir>_export",
    )
    return parser.parse_args()


def load_existing_manifest(manifest_path: str) -> Dict[str, object]:
    if not os.path.isfile(manifest_path):
        return {}
    return load_json(manifest_path)


def default_input_path(explicit_path: Optional[str], default_path: str) -> str:
    if explicit_path:
        return os.path.abspath(explicit_path)
    return default_path


def default_summary_output_dir(
    explicit_summary_output_dir: Optional[str], cluster_output_dir: str
) -> str:
    if explicit_summary_output_dir:
        return os.path.abspath(explicit_summary_output_dir)
    return f"{os.path.abspath(cluster_output_dir)}_summary"


def default_export_output_dir(
    explicit_export_output_dir: Optional[str], cluster_output_dir: str
) -> str:
    if explicit_export_output_dir:
        return os.path.abspath(explicit_export_output_dir)
    return f"{os.path.abspath(cluster_output_dir)}_export"


def build_manifest(
    existing_manifest: Dict[str, object],
    input_labels: str,
    input_distances: str,
    input_summary: str,
    texts_jsonl: Optional[str],
    export_output_paths: Dict[str, str],
    n_texts: int,
    include_text_in_assignments: bool,
) -> Dict[str, object]:
    now = now_utc_iso()
    manifest = dict(existing_manifest)
    manifest.setdefault("created_at_utc", now)
    manifest["updated_at_utc"] = now

    source = dict(manifest.get("source")) if isinstance(manifest.get("source"), dict) else {}
    source["cluster_labels_path"] = input_labels
    source["cluster_distances_path"] = input_distances
    source["cluster_summary_json"] = input_summary
    source["filtered_texts_jsonl"] = texts_jsonl
    manifest["source"] = source

    stats = dict(manifest.get("stats")) if isinstance(manifest.get("stats"), dict) else {}
    stats["n_texts"] = int(n_texts)
    manifest["stats"] = stats

    export_meta = {
        "include_text_in_assignments": bool(include_text_in_assignments),
    }
    manifest["export"] = export_meta

    manifest_outputs = (
        dict(manifest.get("outputs")) if isinstance(manifest.get("outputs"), dict) else {}
    )
    manifest_outputs["cluster_assignments_csv"] = export_output_paths["cluster_assignments_csv"]
    manifest_outputs["cluster_assignments_jsonl"] = export_output_paths["cluster_assignments_jsonl"]
    manifest_outputs["manifest_json"] = export_output_paths["manifest_json"]
    manifest_outputs["cluster_labels_path"] = input_labels
    manifest_outputs["cluster_distances_path"] = input_distances
    manifest_outputs["cluster_summary_json"] = input_summary
    manifest["outputs"] = manifest_outputs
    manifest["pipeline"] = {
        "step": "export",
        "status": "completed",
    }
    return manifest


def main() -> None:
    args = parse_args()

    cluster_output_dir = default_cluster_output_dir(args.output_dir)
    cluster_output_paths = build_output_paths(cluster_output_dir)
    summary_output_dir = default_summary_output_dir(args.summary_output_dir, cluster_output_dir)
    summary_output_paths = build_output_paths(summary_output_dir)
    export_output_dir = ensure_dir(
        default_export_output_dir(args.export_output_dir, cluster_output_dir)
    )
    export_output_paths = build_output_paths(export_output_dir)
    input_labels = default_input_path(args.input_labels, cluster_output_paths["cluster_labels_path"])
    input_distances = default_input_path(
        args.input_distances, cluster_output_paths["cluster_distances_path"]
    )
    input_summary = default_input_path(
        args.input_summary, summary_output_paths["cluster_summary_json"]
    )
    texts_jsonl = default_filtered_texts_jsonl(args.texts_jsonl)

    if not os.path.isfile(input_labels):
        raise FileNotFoundError(f"input_labels not found: {input_labels}")
    if not os.path.isfile(input_distances):
        raise FileNotFoundError(f"input_distances not found: {input_distances}")
    if not os.path.isfile(input_summary):
        raise FileNotFoundError(f"input_summary not found: {input_summary}")
    if args.include_text_in_assignments and not os.path.isfile(texts_jsonl):
        raise FileNotFoundError(f"texts_jsonl not found: {texts_jsonl}")

    labels = np.load(input_labels, mmap_mode="r")
    distances = np.load(input_distances, mmap_mode="r")
    if labels.shape != distances.shape:
        raise RuntimeError(
            f"Shape mismatch: cluster_labels shape={labels.shape}, cluster_distances shape={distances.shape}"
        )

    n_texts = int(labels.shape[0])
    if n_texts == 0:
        raise RuntimeError("No cluster labels available for export. Run 4.cluster.py first.")

    if args.include_text_in_assignments:
        n_text_rows = count_jsonl_records(texts_jsonl)
        if n_text_rows != n_texts:
            raise RuntimeError(
                f"Label/text count mismatch: cluster_labels={n_texts}, filtered_texts={n_text_rows}"
            )

    cluster_labels = load_cluster_labels_from_summary(input_summary)
    cluster_manifest = load_existing_manifest(cluster_output_paths["manifest_json"])
    summary_manifest = load_existing_manifest(summary_output_paths["manifest_json"])
    existing_export_manifest = load_existing_manifest(export_output_paths["manifest_json"])
    if summary_manifest:
        existing_manifest = dict(summary_manifest)
    elif cluster_manifest:
        existing_manifest = dict(cluster_manifest)
    else:
        existing_manifest = dict(existing_export_manifest)
    if "created_at_utc" in existing_export_manifest:
        existing_manifest["created_at_utc"] = existing_export_manifest["created_at_utc"]

    print("Exporting assignments ...")
    write_assignments(
        labels=labels,
        distances=distances,
        cluster_labels=cluster_labels,
        csv_path=export_output_paths["cluster_assignments_csv"],
        jsonl_path=export_output_paths["cluster_assignments_jsonl"],
        include_text=args.include_text_in_assignments,
        texts_path=texts_jsonl,
    )

    manifest = build_manifest(
        existing_manifest=existing_manifest,
        input_labels=input_labels,
        input_distances=input_distances,
        input_summary=input_summary,
        texts_jsonl=texts_jsonl if args.include_text_in_assignments else None,
        export_output_paths=export_output_paths,
        n_texts=n_texts,
        include_text_in_assignments=args.include_text_in_assignments,
    )
    write_json(export_output_paths["manifest_json"], manifest)

    print("Done.")
    print(f"  assignments_csv:   {export_output_paths['cluster_assignments_csv']}")
    print(f"  assignments_jsonl: {export_output_paths['cluster_assignments_jsonl']}")
    print(f"  manifest:          {export_output_paths['manifest_json']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
