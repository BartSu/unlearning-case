"""
Filter locally materialized WikiText JSONL into the format used by clustering scripts.

Default inputs and outputs:
  input:  data/wikitext_raw/raw_texts.jsonl
  output: data/wikitext_filtered/
            filtered_texts.jsonl
            filtered_text_offsets.npy
            filter_manifest.json

The output JSONL stays compatible with downstream scripts by keeping a top-level
"text" field on every record.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


HEADING_RE = re.compile(r"^\s*=+\s.*\s=+\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter raw WikiText JSONL into filtered_texts.jsonl plus offsets."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_raw/raw_texts.jsonl",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="JSON field to read from each raw JSONL line.",
    )
    parser.add_argument("--min_chars", type=int, default=50)
    parser.add_argument(
        "--keep_headings",
        action="store_true",
        help="Keep heading lines '= ... ='. By default they are removed.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on kept rows after filtering.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Default: <data_dir>/wikitext_filtered",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50000,
        help="Print progress every N seen rows. Set to 0 to disable.",
    )
    return parser.parse_args()


def default_input_jsonl(explicit_input_jsonl: Optional[str]) -> str:
    if explicit_input_jsonl:
        return os.path.abspath(explicit_input_jsonl)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)
    return os.path.join(data_dir, "wikitext_raw", "raw_texts.jsonl")


def default_output_dir(explicit_output_dir: Optional[str]) -> str:
    if explicit_output_dir:
        return os.path.abspath(explicit_output_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)
    return os.path.join(data_dir, "wikitext_filtered")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_output_paths(output_dir: str) -> Dict[str, str]:
    return {
        "filtered_texts_jsonl": os.path.join(output_dir, "filtered_texts.jsonl"),
        "filtered_offsets_npy": os.path.join(output_dir, "filtered_text_offsets.npy"),
        "manifest_json": os.path.join(output_dir, "filter_manifest.json"),
    }


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def clean_text(
    text: str,
    min_chars: int,
    keep_headings: bool,
) -> Tuple[Optional[str], Optional[str]]:
    text = text.strip()
    if not text:
        return None, "empty"
    if not keep_headings and HEADING_RE.match(text):
        return None, "heading"
    if len(text) < min_chars:
        return None, "short"
    return text, None


def filter_texts(
    input_jsonl: str,
    text_field: str,
    min_chars: int,
    keep_headings: bool,
    max_rows: Optional[int],
    output_jsonl: str,
    output_offsets: str,
    log_every: int,
) -> Dict[str, int]:
    seen_rows = 0
    kept_rows = 0
    dropped_rows = 0
    dropped_empty_rows = 0
    dropped_heading_rows = 0
    dropped_short_rows = 0
    offsets: List[int] = []

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(
        output_jsonl, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            seen_rows += 1
            record = json.loads(line)
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Expected a JSON object on line {seen_rows} in {input_jsonl}"
                )

            raw_value = record.get(text_field, "")
            cleaned, reason = clean_text(
                "" if raw_value is None else str(raw_value),
                min_chars=min_chars,
                keep_headings=keep_headings,
            )
            if cleaned is None:
                dropped_rows += 1
                if reason == "empty":
                    dropped_empty_rows += 1
                elif reason == "heading":
                    dropped_heading_rows += 1
                elif reason == "short":
                    dropped_short_rows += 1
                continue

            if max_rows is not None and kept_rows >= max_rows:
                break

            output_record = {"text": cleaned}
            if "source_row_id" in record:
                output_record["source_row_id"] = record["source_row_id"]

            offset = fout.tell()
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            offsets.append(offset)
            kept_rows += 1

            if log_every > 0 and seen_rows % log_every == 0:
                print(
                    f"[filter] seen={seen_rows:,} kept={kept_rows:,} dropped={dropped_rows:,}"
                )

    np.save(output_offsets, np.asarray(offsets, dtype=np.uint64))
    return {
        "seen_rows": seen_rows,
        "kept_rows": kept_rows,
        "dropped_rows": dropped_rows,
        "dropped_empty_rows": dropped_empty_rows,
        "dropped_heading_rows": dropped_heading_rows,
        "dropped_short_rows": dropped_short_rows,
    }


def main() -> None:
    args = parse_args()

    input_jsonl = default_input_jsonl(args.input_jsonl)
    if not os.path.isfile(input_jsonl):
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")

    output_dir = ensure_dir(default_output_dir(args.output_dir))
    output_paths = build_output_paths(output_dir)

    print("Filtering texts ...")
    stats = filter_texts(
        input_jsonl=input_jsonl,
        text_field=args.text_field,
        min_chars=args.min_chars,
        keep_headings=args.keep_headings,
        max_rows=args.max_rows,
        output_jsonl=output_paths["filtered_texts_jsonl"],
        output_offsets=output_paths["filtered_offsets_npy"],
        log_every=args.log_every,
    )
    if stats["kept_rows"] == 0:
        raise RuntimeError("No texts available after filtering. Adjust filters and retry.")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_jsonl": input_jsonl,
            "text_field": args.text_field,
        },
        "filter": {
            "min_chars": args.min_chars,
            "keep_headings": args.keep_headings,
            "max_rows": args.max_rows,
        },
        "stats": stats,
        "outputs": output_paths,
    }
    write_json(output_paths["manifest_json"], manifest)

    print("Done.")
    print(f"  filtered: {output_paths['filtered_texts_jsonl']}")
    print(f"  offsets:  {output_paths['filtered_offsets_npy']}")
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
