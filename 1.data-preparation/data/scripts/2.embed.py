"""
Create embeddings from filtered WikiText JSONL for downstream dimensionality reduction.

Default input:
  data/wikitext_filtered/
    filtered_texts.jsonl
    filtered_text_offsets.npy

Default outputs:
  data/wikitext_embeddings/
    embeddings.npy
    run_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

from _hdbscan_pipeline_utils import (
    build_output_paths,
    clear_outputs_after_embed,
    count_jsonl_records,
    count_offsets,
    create_embeddings,
    default_filter_manifest_json,
    default_filtered_offsets_npy,
    default_filtered_texts_jsonl,
    ensure_dir,
    load_json,
    now_utc_iso,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create embeddings from filtered_texts.jsonl."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_filtered/filtered_texts.jsonl",
    )
    parser.add_argument(
        "--input_offsets",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_filtered/filtered_text_offsets.npy",
    )
    parser.add_argument(
        "--filter_manifest",
        type=str,
        default=None,
        help="Optional filter manifest for provenance. Default: <data_dir>/wikitext_filtered/filter_manifest.json",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="auto",
        help="auto|cpu|cuda|mps (forwarded to SentenceTransformer).",
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="L2-normalize embeddings before saving.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_embeddings",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_output_dir(explicit_output_dir: Optional[str]) -> str:
    if explicit_output_dir:
        return os.path.abspath(explicit_output_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)
    return os.path.join(data_dir, "wikitext_embeddings")


def load_existing_manifest(manifest_path: str) -> Dict[str, object]:
    if not os.path.isfile(manifest_path):
        return {}
    return load_json(manifest_path)


def resolve_n_texts(input_jsonl: str, input_offsets: str, offsets_explicit: bool) -> int:
    if os.path.isfile(input_offsets):
        return count_offsets(input_offsets)
    if offsets_explicit:
        raise FileNotFoundError(f"input_offsets not found: {input_offsets}")
    print("filtered_text_offsets.npy not found; counting JSONL lines instead ...")
    return count_jsonl_records(input_jsonl)


def maybe_load_filter_manifest(
    filter_manifest_path: str,
    manifest_explicit: bool,
) -> Optional[Dict[str, object]]:
    if os.path.isfile(filter_manifest_path):
        return load_json(filter_manifest_path)
    if manifest_explicit:
        raise FileNotFoundError(f"filter_manifest not found: {filter_manifest_path}")
    return None


def build_manifest(
    existing_manifest: Dict[str, object],
    input_jsonl: str,
    input_offsets: str,
    filter_manifest_path: str,
    filter_manifest: Optional[Dict[str, object]],
    output_paths: Dict[str, str],
    n_texts: int,
    embedding_meta: Dict[str, object],
) -> Dict[str, object]:
    now = now_utc_iso()
    manifest = dict(existing_manifest)
    manifest.setdefault("created_at_utc", now)
    manifest["updated_at_utc"] = now
    manifest["source"] = {
        "filtered_texts_jsonl": input_jsonl,
        "filtered_offsets_npy": input_offsets if os.path.isfile(input_offsets) else None,
        "filter_manifest_json": filter_manifest_path if filter_manifest is not None else None,
    }
    manifest["stats"] = {"n_texts": int(n_texts)}
    if filter_manifest is not None:
        if isinstance(filter_manifest.get("filter"), dict):
            manifest["filter"] = filter_manifest["filter"]
        else:
            manifest.pop("filter", None)
        if isinstance(filter_manifest.get("stats"), dict):
            manifest["filter_stats"] = filter_manifest["stats"]
        else:
            manifest.pop("filter_stats", None)
    else:
        manifest.pop("filter", None)
        manifest.pop("filter_stats", None)

    manifest["embedding"] = embedding_meta
    manifest["reducer"] = None
    manifest["clustering"] = None
    manifest["outputs"] = output_paths
    manifest["pipeline"] = {
        "step": "embed",
        "status": "completed",
    }
    return manifest


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_jsonl = default_filtered_texts_jsonl(args.input_jsonl)
    if not os.path.isfile(input_jsonl):
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")

    input_offsets = default_filtered_offsets_npy(args.input_offsets)
    filter_manifest_path = default_filter_manifest_json(args.filter_manifest)
    filter_manifest = maybe_load_filter_manifest(
        filter_manifest_path=filter_manifest_path,
        manifest_explicit=args.filter_manifest is not None,
    )
    n_texts = resolve_n_texts(
        input_jsonl=input_jsonl,
        input_offsets=input_offsets,
        offsets_explicit=args.input_offsets is not None,
    )
    if n_texts == 0:
        raise RuntimeError("No texts available for embedding. Run 1.filter.py first.")

    output_dir = ensure_dir(default_output_dir(args.output_dir))
    output_paths = build_output_paths(output_dir)
    existing_manifest = load_existing_manifest(output_paths["manifest_json"])

    clear_outputs_after_embed(output_paths)

    print("Creating embeddings ...")
    print(f"  texts: {n_texts:,}")
    embedding_meta = create_embeddings(
        texts_path=input_jsonl,
        n_texts=n_texts,
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        device=args.embedding_device,
        normalize_embeddings=args.normalize_embeddings,
        embeddings_path=output_paths["embeddings_path"],
    )

    manifest = build_manifest(
        existing_manifest=existing_manifest,
        input_jsonl=input_jsonl,
        input_offsets=input_offsets,
        filter_manifest_path=filter_manifest_path,
        filter_manifest=filter_manifest,
        output_paths=output_paths,
        n_texts=n_texts,
        embedding_meta=embedding_meta,
    )
    write_json(output_paths["manifest_json"], manifest)

    print("Done.")
    print(f"  embeddings: {output_paths['embeddings_path']}")
    print(f"  manifest:   {output_paths['manifest_json']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
