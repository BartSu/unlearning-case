"""
Build per-cluster WikiText triplets from exported cluster assignments.

Default inputs:
  data/wikitext_clusters_hdbscan_export/
    cluster_assignments.csv
    run_manifest.json
  data/wikitext_filtered/
    filtered_texts.jsonl
    filtered_text_offsets.npy

Optional aligned offsets for subset clustering:
  data/wikitext_clusters_hdbscan_summary/
    filtered_text_offsets_subset.npy

Default outputs:
  data/wikitext_hdbscan_triplets/
    triplet_001/train.json
    triplet_001/validation.json
    triplet_001/test.json
    ...
    run_manifest.json

Each triplet corresponds to one cluster:
  - train: sampled from the cluster
  - validation: sampled from the same cluster
  - test: sampled from the same cluster

The three splits are disjoint. By default the noise cluster (-1) is excluded,
and clusters with fewer than forget_size + validation_size + test_size rows are
skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from _hdbscan_pipeline_utils import (
    data_dir,
    default_cluster_output_dir,
    default_filtered_offsets_npy,
    default_filtered_texts_jsonl,
    ensure_dir,
    load_json,
    now_utc_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-cluster triplets from exported cluster assignments."
    )
    parser.add_argument(
        "--assignments_csv",
        type=str,
        default=None,
        help="Default: <export_output_dir>/cluster_assignments.csv",
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
        help=(
            "Default: auto-detect aligned offsets from <export_output_dir>/run_manifest.json, "
            "else <data_dir>/wikitext_filtered/filtered_text_offsets.npy"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Cluster output dir from 4.cluster.py. Default: <data_dir>/wikitext_clusters_hdbscan",
    )
    parser.add_argument(
        "--export_output_dir",
        type=str,
        default=None,
        help="Export output dir from 6.export.py. Default: <output_dir>_export",
    )
    parser.add_argument(
        "--triplet_output_dir",
        type=str,
        default=None,
        help=(
            "Triplet output dir. Default: auto-detect algorithm and write to "
            "<data_dir>/wikitext_hdbscan_triplets or <data_dir>/wikitext_dbscan_triplets."
        ),
    )

    parser.add_argument("--text_id_col", type=str, default="text_id")
    parser.add_argument("--cluster_col", type=str, default="cluster_label")
    parser.add_argument("--domain_col", type=str, default="domain")
    parser.add_argument("--noise_label", type=int, default=-1)
    parser.add_argument(
        "--include_noise",
        action="store_true",
        help="Include the noise cluster (-1) as a regular triplet domain.",
    )

    parser.add_argument(
        "--forget_size",
        type=int,
        default=100,
        help="Number of forget/train samples per cluster.",
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=100,
        help="Number of validation/retain samples per cluster.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of test samples per cluster.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fail_on_small_cluster",
        action="store_true",
        help="Fail instead of skipping clusters smaller than the required total size.",
    )
    return parser.parse_args()


def load_existing_manifest(manifest_path: str) -> Dict[str, object]:
    if not os.path.isfile(manifest_path):
        return {}
    return load_json(manifest_path)


def default_export_output_dir(
    explicit_export_output_dir: Optional[str], cluster_output_dir: str
) -> str:
    if explicit_export_output_dir:
        return os.path.abspath(explicit_export_output_dir)
    return f"{os.path.abspath(cluster_output_dir)}_export"


def default_assignments_csv(
    explicit_assignments_csv: Optional[str], export_output_dir: str
) -> str:
    if explicit_assignments_csv:
        return os.path.abspath(explicit_assignments_csv)
    return os.path.join(export_output_dir, "cluster_assignments.csv")


def infer_default_triplet_output_dir(
    explicit_triplet_output_dir: Optional[str],
    export_manifest: Dict[str, object],
    export_output_dir: str,
) -> str:
    if explicit_triplet_output_dir:
        return os.path.abspath(explicit_triplet_output_dir)

    algorithm = None
    clustering = export_manifest.get("clustering")
    if isinstance(clustering, dict):
        raw_algorithm = clustering.get("algorithm")
        if isinstance(raw_algorithm, str):
            algorithm = raw_algorithm.lower()

    export_dir_name = os.path.basename(os.path.abspath(export_output_dir)).lower()
    root = data_dir()
    if algorithm == "hdbscan" or "hdbscan" in export_dir_name:
        return os.path.join(root, "wikitext_hdbscan_triplets")
    if algorithm == "dbscan" or "dbscan" in export_dir_name:
        return os.path.join(root, "wikitext_dbscan_triplets")
    return os.path.join(root, "wikitext_hdbscan_triplets")


def resolve_offsets_npy(
    explicit_offsets_npy: Optional[str],
    export_manifest: Dict[str, object],
) -> str:
    if explicit_offsets_npy:
        return os.path.abspath(explicit_offsets_npy)

    candidates: List[str] = []
    source = export_manifest.get("source")
    if isinstance(source, dict):
        for key in ("summary_offsets_npy", "filtered_offsets_npy"):
            value = source.get(key)
            if isinstance(value, str):
                candidates.append(os.path.abspath(value))

    outputs = export_manifest.get("outputs")
    if isinstance(outputs, dict):
        value = outputs.get("subset_filtered_offsets_npy")
        if isinstance(value, str):
            candidates.append(os.path.abspath(value))

    fallback_offsets = os.path.abspath(default_filtered_offsets_npy(None))
    candidates.append(fallback_offsets)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate

    return fallback_offsets


def read_assignments(
    assignments_csv: str,
    text_id_col: str,
    cluster_col: str,
    domain_col: str,
    noise_label: int,
    include_noise: bool,
) -> Tuple[Dict[int, List[int]], Dict[int, str], List[int], int]:
    cluster_to_ids: Dict[int, List[int]] = {}
    cluster_to_domain: Dict[int, str] = {}
    used_ids: List[int] = []
    rows_read = 0

    with open(assignments_csv, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            rows_read += 1
            text_id = int(row[text_id_col])
            cluster_id = int(row[cluster_col])
            domain = row.get(domain_col, "") or f"cluster_{cluster_id}"

            if not include_noise and cluster_id == noise_label:
                continue

            if cluster_id not in cluster_to_ids:
                cluster_to_ids[cluster_id] = []
                cluster_to_domain[cluster_id] = domain
            cluster_to_ids[cluster_id].append(text_id)
            used_ids.append(text_id)

    if not cluster_to_ids:
        raise RuntimeError("No usable clusters found in assignments CSV.")

    return cluster_to_ids, cluster_to_domain, used_ids, rows_read


def read_text_by_id(fp, offsets: np.ndarray, text_id: int) -> str:
    fp.seek(int(offsets[text_id]))
    line = fp.readline()
    if not line:
        raise RuntimeError(f"Missing text line for text_id={text_id}")
    record = json.loads(line)
    return record["text"]


def materialize_records(ids: Sequence[int], fp, offsets: np.ndarray) -> List[Dict[str, str]]:
    return [{"text": read_text_by_id(fp, offsets, idx)} for idx in ids]


def sample_cluster_splits(
    ids: Sequence[int],
    forget_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    required_size = forget_size + validation_size + test_size
    if len(ids) < required_size:
        raise RuntimeError(
            f"Need at least {required_size} samples, but cluster only has {len(ids)}."
        )

    shuffled = list(ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    forget_ids = sorted(shuffled[:forget_size])
    validation_start = forget_size
    validation_end = validation_start + validation_size
    validation_ids = sorted(shuffled[validation_start:validation_end])
    test_ids = sorted(shuffled[validation_end : validation_end + test_size])
    return forget_ids, validation_ids, test_ids


def write_records_json(path: str, data: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def build_manifest(
    existing_manifest: Dict[str, object],
    assignments_csv: str,
    texts_jsonl: str,
    offsets_npy: str,
    export_manifest_path: str,
    triplet_output_dir: str,
    manifest_path: str,
    rows_read: int,
    n_texts: int,
    n_used_ids: int,
    args: argparse.Namespace,
    triplets: List[Dict[str, object]],
    skipped_clusters: List[Dict[str, object]],
) -> Dict[str, object]:
    now = now_utc_iso()
    manifest = dict(existing_manifest)
    manifest.setdefault("created_at_utc", now)
    manifest["updated_at_utc"] = now

    source = dict(manifest.get("source")) if isinstance(manifest.get("source"), dict) else {}
    source["cluster_assignments_csv"] = assignments_csv
    source["filtered_texts_jsonl"] = texts_jsonl
    if not isinstance(source.get("filtered_offsets_npy"), str):
        source["filtered_offsets_npy"] = offsets_npy
    source["triplet_offsets_npy"] = offsets_npy
    source["export_manifest_json"] = (
        export_manifest_path if os.path.isfile(export_manifest_path) else None
    )
    manifest["source"] = source

    stats = dict(manifest.get("stats")) if isinstance(manifest.get("stats"), dict) else {}
    stats["n_texts"] = int(n_texts)
    stats["rows_read_assignments"] = int(rows_read)
    stats["n_used_ids"] = int(n_used_ids)
    manifest["stats"] = stats

    manifest["triplet_generation"] = {
        "text_id_col": args.text_id_col,
        "cluster_col": args.cluster_col,
        "domain_col": args.domain_col,
        "include_noise": bool(args.include_noise),
        "noise_label": int(args.noise_label),
        "forget_size": int(args.forget_size),
        "validation_size": int(args.validation_size),
        "test_size": int(args.test_size),
        "required_cluster_size": int(
            args.forget_size + args.validation_size + args.test_size
        ),
        "fail_on_small_cluster": bool(args.fail_on_small_cluster),
        "seed": int(args.seed),
    }

    outputs = dict(manifest.get("outputs")) if isinstance(manifest.get("outputs"), dict) else {}
    outputs["triplet_output_dir"] = triplet_output_dir
    outputs["manifest_json"] = manifest_path
    manifest["outputs"] = outputs

    manifest["n_domains"] = len(triplets)
    manifest["triplets"] = triplets
    manifest["skipped_clusters"] = skipped_clusters
    manifest["pipeline"] = {
        "step": "generate_triplet",
        "status": "completed",
    }
    return manifest


def main() -> None:
    args = parse_args()

    cluster_output_dir = default_cluster_output_dir(args.output_dir)
    export_output_dir = default_export_output_dir(args.export_output_dir, cluster_output_dir)
    assignments_csv = default_assignments_csv(args.assignments_csv, export_output_dir)
    export_manifest_path = os.path.join(export_output_dir, "run_manifest.json")
    export_manifest = load_existing_manifest(export_manifest_path)

    texts_jsonl = default_filtered_texts_jsonl(args.texts_jsonl)
    offsets_npy = resolve_offsets_npy(args.offsets_npy, export_manifest)
    triplet_output_dir = ensure_dir(
        infer_default_triplet_output_dir(
            explicit_triplet_output_dir=args.triplet_output_dir,
            export_manifest=export_manifest,
            export_output_dir=export_output_dir,
        )
    )
    triplet_manifest_path = os.path.join(triplet_output_dir, "run_manifest.json")

    if not os.path.isfile(assignments_csv):
        raise FileNotFoundError(f"assignments_csv not found: {assignments_csv}")
    if not os.path.isfile(texts_jsonl):
        raise FileNotFoundError(f"texts_jsonl not found: {texts_jsonl}")
    if not os.path.isfile(offsets_npy):
        raise FileNotFoundError(f"offsets_npy not found: {offsets_npy}")

    if args.forget_size <= 0 or args.validation_size <= 0 or args.test_size <= 0:
        raise RuntimeError("Require forget_size>0, validation_size>0, test_size>0")

    existing_triplet_manifest = load_existing_manifest(triplet_manifest_path)
    existing_manifest = dict(export_manifest) if export_manifest else dict(existing_triplet_manifest)
    if "created_at_utc" in existing_triplet_manifest:
        existing_manifest["created_at_utc"] = existing_triplet_manifest["created_at_utc"]

    print("Loading assignments ...")
    cluster_to_ids, cluster_to_domain, used_ids, rows_read = read_assignments(
        assignments_csv=assignments_csv,
        text_id_col=args.text_id_col,
        cluster_col=args.cluster_col,
        domain_col=args.domain_col,
        noise_label=args.noise_label,
        include_noise=args.include_noise,
    )

    offsets = np.load(offsets_npy, mmap_mode="r")
    n_texts = int(offsets.shape[0])
    unique_ids = sorted(set(used_ids))

    if unique_ids[-1] >= n_texts:
        raise RuntimeError(
            f"text_id out of range: max text_id={unique_ids[-1]}, offsets rows={n_texts}"
        )

    required_size = args.forget_size + args.validation_size + args.test_size
    triplets: List[Dict[str, object]] = []
    skipped_clusters: List[Dict[str, object]] = []

    with open(texts_jsonl, "r", encoding="utf-8") as fp:
        for i, cluster_id in enumerate(sorted(cluster_to_ids.keys()), start=1):
            triplet_name = f"triplet_{i:03d}"
            cluster_ids = cluster_to_ids[cluster_id]
            cluster_size = len(cluster_ids)

            if cluster_size < required_size:
                skip_info = {
                    "name": triplet_name,
                    "cluster_label": cluster_id,
                    "domain": cluster_to_domain.get(cluster_id, f"cluster_{cluster_id}"),
                    "cluster_size": cluster_size,
                    "required_cluster_size": required_size,
                }
                if args.fail_on_small_cluster:
                    raise RuntimeError(
                        f"{triplet_name} cluster={cluster_id}: "
                        f"cluster_size {cluster_size} < required_size {required_size}"
                    )
                skipped_clusters.append(skip_info)
                print(
                    f"  skip {triplet_name}: cluster={cluster_id} "
                    f"size={cluster_size} < required={required_size}"
                )
                continue

            cluster_seed = args.seed + (cluster_id * 7919)
            forget_ids, validation_ids, test_ids = sample_cluster_splits(
                ids=cluster_ids,
                forget_size=args.forget_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                seed=cluster_seed,
            )

            train_data = materialize_records(forget_ids, fp, offsets)
            validation_data = materialize_records(validation_ids, fp, offsets)
            test_data = materialize_records(test_ids, fp, offsets)

            triplet_dir = os.path.join(triplet_output_dir, triplet_name)
            os.makedirs(triplet_dir, exist_ok=True)

            write_records_json(os.path.join(triplet_dir, "train.json"), train_data)
            write_records_json(os.path.join(triplet_dir, "validation.json"), validation_data)
            write_records_json(os.path.join(triplet_dir, "test.json"), test_data)

            triplets.append(
                {
                    "name": triplet_name,
                    "cluster_label": cluster_id,
                    "domain": cluster_to_domain.get(cluster_id, f"cluster_{cluster_id}"),
                    "cluster_size": cluster_size,
                    "forget_size": len(train_data),
                    "validation_size": len(validation_data),
                    "test_size": len(test_data),
                    "unused_cluster_samples": cluster_size - required_size,
                }
            )
            print(
                f"  {triplet_name}: cluster={cluster_id} "
                f"train={len(train_data)} validation={len(validation_data)} "
                f"test={len(test_data)}"
            )

    manifest = build_manifest(
        existing_manifest=existing_manifest,
        assignments_csv=assignments_csv,
        texts_jsonl=texts_jsonl,
        offsets_npy=offsets_npy,
        export_manifest_path=export_manifest_path,
        triplet_output_dir=triplet_output_dir,
        manifest_path=triplet_manifest_path,
        rows_read=rows_read,
        n_texts=n_texts,
        n_used_ids=len(unique_ids),
        args=args,
        triplets=triplets,
        skipped_clusters=skipped_clusters,
    )
    write_json(triplet_manifest_path, manifest)

    print("\nDone.")
    print(f"  triplet_output_dir: {triplet_output_dir}")
    print(f"  offsets_npy:        {offsets_npy}")
    print(f"  domains_kept:       {manifest['n_domains']}")
    print(f"  clusters_skipped:   {len(skipped_clusters)}")
    print(f"  manifest:           {triplet_manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
