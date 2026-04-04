"""
Build a combined label subset from triplet test splits.

Default input:
  data/wikitext_hdbscan_triplets/
    triplet_001/test.json
    triplet_002/test.json
    ...
    run_manifest.json

Default output:
  data/wikitext_label/
    label.json
    run_manifest.json

The output label.json contains the combined deterministic samples drawn from
every selected triplet's test split. By default, 10 texts are selected per
triplet and merged into one file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from _hdbscan_pipeline_utils import data_dir, ensure_dir, load_json, now_utc_iso, write_json


DEFAULT_TRIPLET_DIR_CANDIDATES = (
    "wikitext_hdbscan_triplets",
    "wikitext_dbscan_triplets",
)
DEFAULT_LABEL_DIR_NAME = "wikitext_label"
TEST_FILENAME = "test.json"
LABEL_FILENAME = "label.json"
MANIFEST_FILENAME = "run_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and combine label subsets from triplet test splits."
    )
    parser.add_argument(
        "--triplet_input_dir",
        type=str,
        default=None,
        help=(
            "Triplet root dir. Default: auto-detect "
            "<data_dir>/wikitext_hdbscan_triplets, then <data_dir>/wikitext_dbscan_triplets."
        ),
    )
    parser.add_argument(
        "--label_output_dir",
        type=str,
        default=None,
        help="Label output dir. Default: <data_dir>/wikitext_label",
    )
    parser.add_argument(
        "--label_size",
        type=int,
        default=10,
        help="Number of texts to sample from each triplet test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used for deterministic per-triplet sampling.",
    )
    parser.add_argument(
        "--triplets",
        type=str,
        default=None,
        help=(
            'Specific triplets to sample, e.g. "triplet_001 triplet_021" or '
            '"triplet_001,triplet_021". Overrides --start/--end.'
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First triplet index for range-based selection (default: 1).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last triplet index for range-based selection (default: all).",
    )
    return parser.parse_args()


def canonicalize_triplet_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    if text.isdigit():
        return f"triplet_{int(text):03d}"

    for pattern in (r"triplet_(\d+)", r"triple_(\d+)"):
        match = re.search(pattern, text)
        if match:
            return f"triplet_{int(match.group(1)):03d}"

    return text


def parse_triplet_selection(raw_triplets: Optional[str]) -> Optional[List[str]]:
    if not raw_triplets:
        return None

    selected: List[str] = []
    seen = set()
    for chunk in str(raw_triplets).split(","):
        for item in chunk.split():
            triplet_name = canonicalize_triplet_name(item)
            if triplet_name and triplet_name not in seen:
                selected.append(triplet_name)
                seen.add(triplet_name)
    return selected or None


def triplet_sort_key(value: str) -> Tuple[int, str]:
    match = re.search(r"triplet_(\d+)", value)
    if match:
        return int(match.group(1)), value
    return 10**9, value


def extract_triplet_index(value: str) -> int:
    canonical = canonicalize_triplet_name(value)
    match = re.search(r"triplet_(\d+)", canonical)
    if not match:
        raise RuntimeError(f"Could not parse triplet index from: {value}")
    return int(match.group(1))


def resolve_triplet_input_dir(explicit_path: Optional[str]) -> str:
    if explicit_path:
        triplet_input_dir = os.path.abspath(explicit_path)
        if not os.path.isdir(triplet_input_dir):
            raise FileNotFoundError(f"triplet_input_dir not found: {triplet_input_dir}")
        return triplet_input_dir

    root = data_dir()
    for dirname in DEFAULT_TRIPLET_DIR_CANDIDATES:
        candidate = os.path.join(root, dirname)
        if os.path.isdir(candidate):
            return candidate

    tried = ", ".join(os.path.join(root, dirname) for dirname in DEFAULT_TRIPLET_DIR_CANDIDATES)
    raise FileNotFoundError(
        f"Could not auto-detect triplet_input_dir. Tried: {tried}. "
        "Pass --triplet_input_dir explicitly."
    )


def resolve_label_output_dir(explicit_path: Optional[str]) -> str:
    if explicit_path:
        return os.path.abspath(explicit_path)
    return os.path.join(data_dir(), DEFAULT_LABEL_DIR_NAME)


def load_existing_manifest(manifest_path: str) -> Dict[str, object]:
    if not os.path.isfile(manifest_path):
        return {}
    return load_json(manifest_path)


def build_triplet_meta_index(manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    raw_triplets = manifest.get("triplets")
    if not isinstance(raw_triplets, list):
        return index

    for entry in raw_triplets:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str):
            index[canonicalize_triplet_name(name)] = dict(entry)
    return index


def discover_triplets(
    triplet_input_dir: str,
    start: int,
    end: Optional[int],
    selected_triplets: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    triplets: Dict[str, str] = {}
    for entry in os.listdir(triplet_input_dir):
        triplet_name = canonicalize_triplet_name(entry)
        triplet_dir = os.path.join(triplet_input_dir, entry)
        test_path = os.path.join(triplet_dir, TEST_FILENAME)
        if not os.path.isdir(triplet_dir) or not os.path.isfile(test_path):
            continue
        if not triplet_name.startswith("triplet_"):
            continue
        triplets[triplet_name] = triplet_dir

    if selected_triplets:
        missing = [name for name in selected_triplets if name not in triplets]
        if missing:
            raise FileNotFoundError(
                f"Requested triplets not found under {triplet_input_dir}: {', '.join(missing)}"
            )
        return [(name, triplets[name]) for name in selected_triplets]

    upper = end or 9999
    discovered = []
    for name in sorted(triplets.keys(), key=triplet_sort_key):
        triplet_idx = extract_triplet_index(name)
        if start <= triplet_idx <= upper:
            discovered.append((name, triplets[name]))
    return discovered


def read_records_json(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list in {path}")

    records: List[Dict[str, str]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise RuntimeError(
                f"Expected item {idx} in {path} to be an object with a string 'text' field"
            )
        records.append({"text": item["text"]})
    return records


def write_records_json(path: str, data: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def sample_label_records(
    records: Sequence[Dict[str, str]],
    label_size: int,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[int]]:
    if len(records) < label_size:
        raise RuntimeError(
            f"Need at least {label_size} test records, but found only {len(records)}."
        )

    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(records)), label_size))
    sampled_records = [dict(records[idx]) for idx in sampled_indices]
    return sampled_records, sampled_indices


def build_manifest(
    existing_manifest: Dict[str, object],
    triplet_input_dir: str,
    triplet_manifest_path: str,
    label_output_dir: str,
    label_json_path: str,
    output_manifest_path: str,
    args: argparse.Namespace,
    written_triplets: List[Dict[str, object]],
) -> Dict[str, object]:
    now = now_utc_iso()
    manifest = dict(existing_manifest)
    manifest.setdefault("created_at_utc", now)
    manifest["updated_at_utc"] = now

    manifest["source"] = {
        "triplet_input_dir": triplet_input_dir,
        "triplet_manifest_json": triplet_manifest_path if os.path.isfile(triplet_manifest_path) else None,
    }
    manifest["label_generation"] = {
        "label_size": int(args.label_size),
        "seed": int(args.seed),
    }
    manifest["outputs"] = {
        "label_output_dir": label_output_dir,
        "label_json": label_json_path,
        "manifest_json": output_manifest_path,
    }
    manifest["stats"] = {
        "n_triplets": len(written_triplets),
        "total_label_texts": sum(int(item["label_size"]) for item in written_triplets),
    }
    manifest["triplets"] = written_triplets
    manifest["pipeline"] = {
        "step": "label",
        "status": "completed",
    }
    return manifest


def main() -> None:
    args = parse_args()
    args.triplet_list = parse_triplet_selection(args.triplets)

    if args.label_size <= 0:
        raise RuntimeError("Require label_size > 0")

    triplet_input_dir = resolve_triplet_input_dir(args.triplet_input_dir)
    label_output_dir = ensure_dir(resolve_label_output_dir(args.label_output_dir))
    label_json_path = os.path.join(label_output_dir, LABEL_FILENAME)
    triplet_manifest_path = os.path.join(triplet_input_dir, MANIFEST_FILENAME)
    output_manifest_path = os.path.join(label_output_dir, MANIFEST_FILENAME)

    existing_output_manifest = load_existing_manifest(output_manifest_path)
    triplet_manifest = load_existing_manifest(triplet_manifest_path)
    triplet_meta_index = build_triplet_meta_index(triplet_manifest)

    triplets = discover_triplets(
        triplet_input_dir=triplet_input_dir,
        start=args.start,
        end=args.end,
        selected_triplets=args.triplet_list,
    )
    if not triplets:
        raise RuntimeError("No triplets found for the requested selection.")

    selection_label = (
        f"selected {', '.join(args.triplet_list)}"
        if args.triplet_list
        else f"range {args.start}-{args.end or 'all'}"
    )

    print("Sampling label sets from triplet test splits ...")
    print(f"  triplet_input_dir: {triplet_input_dir}")
    print(f"  label_output_dir:  {label_output_dir}")
    print(f"  label_json:        {label_json_path}")
    print(f"  label_size:        {args.label_size}")
    print(f"  triplets:          {len(triplets)} ({selection_label})")

    combined_label_records: List[Dict[str, object]] = []
    written_triplets: List[Dict[str, object]] = []
    for i, (triplet_name, triplet_dir) in enumerate(triplets, start=1):
        test_path = os.path.join(triplet_dir, TEST_FILENAME)
        test_records = read_records_json(test_path)
        triplet_seed = args.seed + (extract_triplet_index(triplet_name) * 7919)
        label_records, sampled_indices = sample_label_records(
            records=test_records,
            label_size=args.label_size,
            seed=triplet_seed,
        )

        source_meta = triplet_meta_index.get(triplet_name, {})
        label_record_offset_start = len(combined_label_records)
        for source_test_index, label_record in zip(sampled_indices, label_records):
            combined_label_records.append(
                {
                    "triplet": triplet_name,
                    "cluster_label": source_meta.get("cluster_label"),
                    "domain": source_meta.get("domain"),
                    "domain_triplet_index": source_meta.get("domain_triplet_index"),
                    "source_test_index": int(source_test_index),
                    "text": label_record["text"],
                }
            )
        label_record_offset_end = len(combined_label_records) - 1

        written_triplets.append(
            {
                "name": triplet_name,
                "cluster_label": source_meta.get("cluster_label"),
                "domain": source_meta.get("domain"),
                "domain_triplet_index": source_meta.get("domain_triplet_index"),
                "source_test_size": len(test_records),
                "label_size": len(label_records),
                "selected_test_indices": sampled_indices,
                "label_record_offset_start": label_record_offset_start,
                "label_record_offset_end": label_record_offset_end,
            }
        )
        print(
            f"  [{i:3d}/{len(triplets)}] {triplet_name}: "
            f"test={len(test_records)} label={len(label_records)}"
        )

    write_records_json(label_json_path, combined_label_records)

    manifest = build_manifest(
        existing_manifest=existing_output_manifest,
        triplet_input_dir=triplet_input_dir,
        triplet_manifest_path=triplet_manifest_path,
        label_output_dir=label_output_dir,
        label_json_path=label_json_path,
        output_manifest_path=output_manifest_path,
        args=args,
        written_triplets=written_triplets,
    )
    write_json(output_manifest_path, manifest)

    print("\nDone.")
    print(f"  label_output_dir: {label_output_dir}")
    print(f"  label_json:       {label_json_path}")
    print(f"  label_records:    {len(combined_label_records)}")
    print(f"  triplets_written: {manifest['stats']['n_triplets']}")
    print(f"  manifest:         {output_manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
