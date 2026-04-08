"""
Download raw WikiText rows from Hugging Face and materialize them to local JSONL.

Default outputs:
  data/wikitext_raw/
    raw_texts.jsonl
    raw_text_offsets.npy
    download_manifest.json

The JSONL records use a normalized schema:
  {"source_row_id": 0, "text": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a WikiText split and materialize raw rows to JSONL."
    )
    parser.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Dataset column to materialize into the normalized JSONL 'text' field.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows written to raw_texts.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_raw",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50000,
        help="Print progress every N written rows. Set to 0 to disable.",
    )
    return parser.parse_args()


def default_output_dir(explicit_output_dir: Optional[str]) -> str:
    if explicit_output_dir:
        return os.path.abspath(explicit_output_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)
    return os.path.join(data_dir, "wikitext_raw")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_output_paths(output_dir: str) -> Dict[str, str]:
    return {
        "raw_texts_jsonl": os.path.join(output_dir, "raw_texts.jsonl"),
        "raw_offsets_npy": os.path.join(output_dir, "raw_text_offsets.npy"),
        "manifest_json": os.path.join(output_dir, "download_manifest.json"),
    }


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def materialize_raw_texts(
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_field: str,
    max_rows: Optional[int],
    output_jsonl: str,
    output_offsets: str,
    log_every: int,
) -> Dict[str, int]:
    ds = load_dataset(dataset_name, dataset_config, split=split)

    rows_written = 0
    empty_text_rows = 0
    offsets: List[int] = []

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for row_id, row in enumerate(ds):
            if max_rows is not None and rows_written >= max_rows:
                break

            raw_value = row.get(text_field, "")
            text = "" if raw_value is None else str(raw_value)
            if not text.strip():
                empty_text_rows += 1

            offset = fout.tell()
            fout.write(
                json.dumps(
                    {"source_row_id": row_id, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
            offsets.append(offset)
            rows_written += 1

            if log_every > 0 and rows_written % log_every == 0:
                print(f"[download] written {rows_written:,} rows")

    np.save(output_offsets, np.asarray(offsets, dtype=np.uint64))
    return {
        "rows_written": rows_written,
        "empty_text_rows": empty_text_rows,
    }


def main() -> None:
    args = parse_args()

    output_dir = ensure_dir(default_output_dir(args.output_dir))
    output_paths = build_output_paths(output_dir)

    print("Downloading and materializing raw texts ...")
    stats = materialize_raw_texts(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        max_rows=args.max_rows,
        output_jsonl=output_paths["raw_texts_jsonl"],
        output_offsets=output_paths["raw_offsets_npy"],
        log_every=args.log_every,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "text_field": args.text_field,
            "max_rows": args.max_rows,
        },
        "stats": stats,
        "outputs": output_paths,
    }
    write_json(output_paths["manifest_json"], manifest)

    print("Done.")
    print(f"  raw_texts: {output_paths['raw_texts_jsonl']}")
    print(f"  offsets:   {output_paths['raw_offsets_npy']}")
    print(f"  manifest:  {output_paths['manifest_json']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
