"""
Evaluate WikiText triplet unlearning models with loss/ppl metrics only.

Output format is JSON only (no CSV):
  - baseline JSON: base model loss/ppl on each triplet's available splits
  - metrics JSON: cross-eval rows for each (unlearn model triplet, eval triplet)
    pair, with base vs unlearn loss/ppl on available splits

Supports both the original text-only triplet format (train/validation/test.json)
and the QA-enriched format (train.json with text + qa fields).

Pipeline:
  1. Compute baselines:
     python eval_wikitext_perplexity.py --baseline --end 50 --resume

  2. Compute cross-eval metrics:
     python eval_wikitext_perplexity.py --saves_dir ./saves/wikitext_unlearn --end 50 --resume

  3. With QA data (qa_prompt perplexity):
     python eval_wikitext_perplexity.py --baseline --data_dir ../data/wikitext_hdbscan_triplets_qa --text_field qa_prompt --resume
"""

import json
import argparse
import re
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WIKITEXT_DIR_CANDIDATES = (
    PROJECT_ROOT / "1.data-preparation" / "data" / "wikitext_hdbscan_triplets",
    PROJECT_ROOT / "1.data-preparation" / "data" / "wikitext_hdbscan_triplets_qa",
    PROJECT_ROOT / "1.data-preparation" / "data" / "wikitext",
)
OUT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASELINE_FILE = "wikitext_baseline.json"
DEFAULT_METRICS_FILE = "wikitext_cross_metrics.json"
TRIPLET_SPLIT_FILENAMES = {
    "train": ("train.json",),
    "validation": ("validation.json", "val.json"),
    "test": ("test.json",),
}
TRIPLET_SPLITS = tuple(TRIPLET_SPLIT_FILENAMES.keys())
BASELINE_SCHEMA_VERSION = 2
METRICS_SCHEMA_VERSION = 3


def load_texts(json_path, text_field="text"):
    with open(json_path) as f:
        return [item[text_field] for item in json.load(f)]


@torch.no_grad()
def compute_avg_loss(model, tokenizer, texts, max_length=512, batch_size=4):
    """Average per-token cross-entropy loss and perplexity."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        labels = encodings["input_ids"].clone()
        labels[encodings["attention_mask"] == 0] = -100
        outputs = model(**encodings, labels=labels)

        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return avg_loss, float(np.exp(avg_loss))


def load_model(model_path, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto",
    )
    return model, tokenizer


def canonicalize_triplet_name(value):
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


def parse_triplet_selection(raw_triplets):
    if not raw_triplets:
        return None

    selected = []
    seen = set()
    for chunk in str(raw_triplets).split(","):
        for item in chunk.split():
            triplet_name = canonicalize_triplet_name(item)
            if triplet_name and triplet_name not in seen:
                selected.append(triplet_name)
                seen.add(triplet_name)

    return selected or None


def describe_triplet_selection(selected_triplets, start, end):
    if selected_triplets:
        return f"selected {', '.join(selected_triplets)}"
    return f"range {start}-{end or 'all'}"


def get_triplet_dirs(data_dir, start=1, end=None, selected_triplets=None):
    dirs = {
        canonicalize_triplet_name(d.name): d
        for d in sorted(data_dir.glob("triplet_*"))
        if d.is_dir()
    }

    if selected_triplets:
        missing = [name for name in selected_triplets if name not in dirs]
        if missing:
            raise FileNotFoundError(
                f"Requested triplets not found under {data_dir}: {', '.join(missing)}"
            )
        return [dirs[name] for name in selected_triplets]

    selected = []
    upper = end or 9999
    for triplet_name, triplet_dir in dirs.items():
        try:
            triplet_idx = int(triplet_name.split("_")[1])
        except (ValueError, IndexError, AttributeError):
            continue
        if start <= triplet_idx <= upper:
            selected.append(triplet_dir)
    return selected


def _save_baseline(baseline, path):
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)


def _save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def resolve_baseline_path(args):
    if args.baseline_file:
        return Path(args.baseline_file)
    return OUT_DIR / BASELINE_FILE


def resolve_data_dir(data_dir_arg):
    if data_dir_arg:
        data_dir = Path(data_dir_arg)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"WikiText data dir not found: {data_dir}")
        return data_dir

    for candidate in WIKITEXT_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate

    tried = ", ".join(str(path) for path in WIKITEXT_DIR_CANDIDATES)
    raise FileNotFoundError(
        f"Could not auto-detect WikiText triplet dir. Tried: {tried}. "
        "Pass --data_dir explicitly."
    )


def resolve_triplet_split_path(triplet_dir, split_name):
    for filename in TRIPLET_SPLIT_FILENAMES[split_name]:
        path = triplet_dir / filename
        if path.is_file():
            return path

    expected = ", ".join(TRIPLET_SPLIT_FILENAMES[split_name])
    raise FileNotFoundError(
        f"Missing {split_name} split under {triplet_dir}. Expected one of: {expected}"
    )


def discover_triplet_splits(triplet_dir):
    """Return split names that have data files in this triplet directory."""
    available = []
    for split_name, filenames in TRIPLET_SPLIT_FILENAMES.items():
        if any((triplet_dir / fn).is_file() for fn in filenames):
            available.append(split_name)
    return tuple(available) if available else ("train",)


def load_triplet_texts(triplet_dir, text_field="text"):
    splits = discover_triplet_splits(triplet_dir)
    return {
        split_name: load_texts(
            resolve_triplet_split_path(triplet_dir, split_name),
            text_field=text_field,
        )
        for split_name in splits
    }


def metric_dict(loss, ppl):
    return {"loss": round(loss, 6), "ppl": round(ppl, 2)}


def normalize_triplet_metrics(metrics):
    normalized = dict(metrics) if isinstance(metrics, dict) else {}
    if "train" not in normalized and isinstance(normalized.get("forget"), dict):
        normalized["train"] = normalized["forget"]
    if "validation" not in normalized and isinstance(normalized.get("retain"), dict):
        normalized["validation"] = normalized["retain"]
    return normalized


def has_complete_triplet_metrics(metrics, available_splits=None):
    if available_splits is None:
        available_splits = TRIPLET_SPLITS
    normalized = normalize_triplet_metrics(metrics)
    return all(isinstance(normalized.get(split_name), dict) for split_name in available_splits)


def evaluate_triplet_splits(model, tokenizer, texts_by_split, max_length, batch_size):
    return {
        split_name: metric_dict(
            *compute_avg_loss(
                model, tokenizer, texts, max_length=max_length, batch_size=batch_size
            )
        )
        for split_name, texts in texts_by_split.items()
    }


def extract_triplet_id(dirname):
    """Extract canonical 'triplet_NNN' from a model directory name."""
    return canonicalize_triplet_name(dirname)


def ensure_baseline_compatible(baseline, args, data_dir):
    cached_model = baseline.get("model")
    if cached_model and cached_model != args.base_model:
        raise ValueError(
            f"Baseline cache was created for {cached_model}, not {args.base_model}. "
            "Use a different --baseline_file or delete the stale cache."
        )
    cached_data_dir = baseline.get("data_dir")
    if cached_data_dir and cached_data_dir != str(data_dir):
        raise ValueError(
            f"Baseline cache points to {cached_data_dir}, not {data_dir}. "
            "Use a different --baseline_file or delete the stale cache."
        )


# ── Phase 1: Baselines ──────────────────────────────────────────────────────

def compute_baselines(args):
    """Base model loss/ppl on each triplet's available splits."""
    baseline_path = resolve_baseline_path(args)
    data_dir = resolve_data_dir(args.data_dir)
    selected_triplets = getattr(args, "eval_triplet_list", None)

    baseline = {}
    if args.resume and baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline["triplets"] = {
            name: normalize_triplet_metrics(metrics)
            for name, metrics in baseline.get("triplets", {}).items()
        }
        n_cached = sum(
            1 for metrics in baseline.get("triplets", {}).values()
            if isinstance(metrics, dict) and metrics
        )
        print(f"Resuming from existing baseline ({n_cached} triplets cached)")

    ensure_baseline_compatible(baseline, args, data_dir)

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = load_model(args.base_model)

    baseline["schema_version"] = BASELINE_SCHEMA_VERSION
    baseline["evaluation_scope"] = "per_triplet"
    baseline["model"] = args.base_model
    baseline["data_dir"] = str(data_dir)
    baseline["text_field"] = args.text_field
    baseline.setdefault("triplets", {})

    if args.test_path and "shared_test" not in baseline:
        test_texts = load_texts(args.test_path, text_field=args.text_field)
        print(f"\nBaseline on shared test set ({len(test_texts)} texts) ...")
        loss, ppl = compute_avg_loss(
            model, tokenizer, test_texts, args.max_length, args.batch_size
        )
        baseline["shared_test"] = metric_dict(loss, ppl)
        print(f"  shared_test: loss={loss:.6f}  ppl={ppl:.2f}")
        _save_baseline(baseline, baseline_path)

    triplet_dirs = get_triplet_dirs(
        data_dir,
        start=args.start,
        end=args.end,
        selected_triplets=selected_triplets,
    )
    remaining = [
        d for d in triplet_dirs
        if not has_complete_triplet_metrics(
            baseline["triplets"].get(d.name),
            available_splits=discover_triplet_splits(d),
        )
    ]
    selection_label = describe_triplet_selection(selected_triplets, args.start, args.end)
    print(f"\nPer-triplet baselines: {len(remaining)} remaining "
          f"(of {len(triplet_dirs)} total, {selection_label})")

    for i, tdir in enumerate(remaining, 1):
        name = tdir.name
        texts_by_split = load_triplet_texts(tdir, text_field=args.text_field)
        baseline["triplets"][name] = evaluate_triplet_splits(
            model,
            tokenizer,
            texts_by_split,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        split_parts = "  ".join(
            f"{s}={baseline['triplets'][name][s]['ppl']:.2f}"
            for s in baseline["triplets"][name]
        )
        print(f"  [{i:3d}/{len(remaining)}] {name}: {split_parts}")

        if i % 5 == 0 or i == len(remaining):
            _save_baseline(baseline, baseline_path)

    _save_baseline(baseline, baseline_path)
    print(f"\nBaseline saved to {baseline_path}")

    del model
    torch.cuda.empty_cache()
    return baseline


# ── Phase 2: Unlearn metrics (JSON only) ────────────────────────────────────

def compute_labels(args):
    """Compute cross-eval loss/ppl metrics for each model/eval triplet pair."""
    baseline_path = resolve_baseline_path(args)
    data_dir = resolve_data_dir(args.data_dir)
    saves = Path(args.saves_dir)
    output_path = Path(args.output) if args.output else OUT_DIR / DEFAULT_METRICS_FILE
    eval_triplets = getattr(args, "eval_triplet_list", None)
    model_triplets = getattr(args, "model_triplet_list", None)

    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline["triplets"] = {
            name: normalize_triplet_metrics(metrics)
            for name, metrics in baseline.get("triplets", {}).items()
        }
        print(f"Loaded baselines ({len(baseline.get('triplets', {}))} triplets)")
    else:
        print("No baseline cache found, computing baselines first ...")
        baseline = compute_baselines(args)

    ensure_baseline_compatible(baseline, args, data_dir)

    eval_triplet_dirs = get_triplet_dirs(
        data_dir,
        start=args.start,
        end=args.end,
        selected_triplets=eval_triplets,
    )
    if not eval_triplet_dirs:
        print("\nNo evaluation triplets found. Check --data_dir and triplet selection.")
        return

    missing_baselines = [
        tdir.name
        for tdir in eval_triplet_dirs
        if not has_complete_triplet_metrics(
            baseline.get("triplets", {}).get(tdir.name),
            available_splits=discover_triplet_splits(tdir),
        )
    ]
    if missing_baselines:
        print(
            f"Missing baselines for {len(missing_baselines)} triplets, computing them now ..."
        )
        baseline = compute_baselines(args)
        baseline["triplets"] = {
            name: normalize_triplet_metrics(metrics)
            for name, metrics in baseline.get("triplets", {}).items()
        }

    eval_triplet_names = [tdir.name for tdir in eval_triplet_dirs]
    eval_triplet_texts = {
        tdir.name: load_triplet_texts(tdir, text_field=args.text_field)
        for tdir in eval_triplet_dirs
    }

    if isinstance(baseline.get("shared_test"), dict):
        print(
            "Shared test: "
            f"loss={baseline['shared_test']['loss']:.6f}  "
            f"ppl={baseline['shared_test']['ppl']:.2f}"
        )

    if (saves / "config.json").exists():
        # Allow passing a single model directory directly.
        model_dirs = [saves]
    else:
        model_dirs = sorted(saves.glob("wikitext_*_triplet_*_GradAscent"))
        if not model_dirs:
            model_dirs = sorted(saves.glob("wikitext_*_triple_*_GradAscent"))
        if not model_dirs:
            model_dirs = sorted(list(saves.glob("*triplet*")) + list(saves.glob("*triple*")))

    if model_triplets:
        selected_set = set(model_triplets)
        model_dirs = [d for d in model_dirs if extract_triplet_id(d.name) in selected_set]
        found_triplets = {extract_triplet_id(d.name) for d in model_dirs}
        missing = [name for name in model_triplets if name not in found_triplets]
        if missing:
            raise FileNotFoundError(
                f"Requested triplet models not found under {saves}: {', '.join(missing)}"
            )
    elif args.start or args.end:
        lo, hi = args.start, args.end or 9999

        def _in_range(d):
            tid = extract_triplet_id(d.name)
            try:
                return lo <= int(tid.split("_")[1]) <= hi
            except (ValueError, IndexError):
                return False

        model_dirs = [d for d in model_dirs if _in_range(d)]

    if not model_dirs:
        print("\nNo unlearned models found. Check --saves_dir and triplet selection.")
        return

    model_triplet_names = list(dict.fromkeys(extract_triplet_id(d.name) for d in model_dirs))
    model_selection_label = describe_triplet_selection(model_triplets, args.start, args.end)
    eval_selection_label = describe_triplet_selection(eval_triplets, args.start, args.end)
    total_pairs = len(model_dirs) * len(eval_triplet_dirs)

    print(f"Found {len(model_dirs)} unlearned models ({model_selection_label})")
    print(f"Evaluating on {len(eval_triplet_dirs)} triplets ({eval_selection_label})")
    print(f"Total model/eval pairs: {total_pairs}\n")

    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluation_scope": "cross_triplet",
        "baseline_file": str(baseline_path),
        "base_model": baseline.get("model", args.base_model),
        "data_dir": str(data_dir),
        "text_field": args.text_field,
        "saves_dir": str(saves),
        "range": {"start": args.start, "end": args.end},
        "eval_triplets": eval_triplet_names,
        "model_triplets": model_triplet_names,
        "results": [],
    }

    done_pairs = set()
    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if (
                isinstance(existing, dict)
                and existing.get("schema_version") == METRICS_SCHEMA_VERSION
                and existing.get("evaluation_scope") == "cross_triplet"
                and isinstance(existing.get("results"), list)
                and existing.get("baseline_file") == str(baseline_path)
                and existing.get("data_dir") == str(data_dir)
                and existing.get("saves_dir") == str(saves)
            ):
                metrics = existing
                done_pairs = {
                    (row.get("model_triplet"), row.get("eval_triplet"))
                    for row in metrics["results"]
                    if (
                        row.get("model_triplet")
                        and row.get("eval_triplet")
                        and isinstance(row.get("base"), dict)
                        and isinstance(row.get("unlearn"), dict)
                    )
                }
                print(f"Resuming: {len(done_pairs)} model/eval pairs already evaluated")
            else:
                print("Existing output uses a different metrics format, starting fresh")
        except json.JSONDecodeError:
            print("Existing output is not valid JSON, starting fresh")

    metrics["schema_version"] = METRICS_SCHEMA_VERSION
    metrics["evaluation_scope"] = "cross_triplet"
    metrics["baseline_file"] = str(baseline_path)
    metrics["base_model"] = baseline.get("model", args.base_model)
    metrics["data_dir"] = str(data_dir)
    metrics["text_field"] = args.text_field
    metrics["saves_dir"] = str(saves)
    metrics["range"] = {"start": args.start, "end": args.end}
    metrics["eval_triplets"] = eval_triplet_names
    metrics["model_triplets"] = model_triplet_names

    shared_test_texts = load_texts(args.test_path, text_field=args.text_field) if args.test_path else None

    progress = tqdm(total=total_pairs, desc="Evaluating pairs")
    for mdir in model_dirs:
        model_triplet = extract_triplet_id(mdir.name)
        model, tokenizer = load_model(str(mdir))

        shared_test_metric = None
        if shared_test_texts is not None and isinstance(baseline.get("shared_test"), dict):
            shared_test_loss, shared_test_ppl = compute_avg_loss(
                model,
                tokenizer,
                shared_test_texts,
                args.max_length,
                args.batch_size,
            )
            shared_test_metric = metric_dict(shared_test_loss, shared_test_ppl)

        new_pairs_for_model = 0
        for eval_triplet in eval_triplet_names:
            pair_key = (model_triplet, eval_triplet)
            if pair_key in done_pairs:
                progress.update(1)
                continue

            triplet_base = baseline.get("triplets", {}).get(eval_triplet)
            eval_splits = tuple(eval_triplet_texts[eval_triplet].keys())
            if not has_complete_triplet_metrics(triplet_base, available_splits=eval_splits):
                print(f"  SKIP {model_triplet} x {eval_triplet}: no baseline cached")
                progress.update(1)
                continue
            triplet_base = normalize_triplet_metrics(triplet_base)

            unlearn_metrics = evaluate_triplet_splits(
                model,
                tokenizer,
                eval_triplet_texts[eval_triplet],
                max_length=args.max_length,
                batch_size=args.batch_size,
            )

            base_metrics = {
                split_name: triplet_base[split_name]
                for split_name in eval_splits
                if split_name in triplet_base
            }
            if shared_test_metric is not None:
                base_metrics["shared_test"] = baseline["shared_test"]
                unlearn_metrics["shared_test"] = shared_test_metric

            metrics["results"].append({
                "model_triplet": model_triplet,
                "eval_triplet": eval_triplet,
                "model_dir": str(mdir),
                "base": base_metrics,
                "unlearn": unlearn_metrics,
            })
            new_pairs_for_model += 1
            progress.update(1)

        print(f"  {model_triplet}: evaluated {new_pairs_for_model} new triplets")
        del model
        torch.cuda.empty_cache()
        _save_metrics(metrics, output_path)

    progress.close()

    if metrics["results"]:
        _save_metrics(metrics, output_path)
        covered_models = {
            row.get("model_triplet")
            for row in metrics["results"]
            if row.get("model_triplet")
        }
        covered_evals = {
            row.get("eval_triplet")
            for row in metrics["results"]
            if row.get("eval_triplet")
        }
        print(f"\nMetrics saved to {output_path}  ({len(metrics['results'])} rows)")
        print(
            f"Cross-eval coverage: {len(covered_models)} models x "
            f"{len(covered_evals)} eval triplets"
        )
    else:
        print("\nNo models evaluated. Check --saves_dir path and triplet range.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WikiText triplet unlearning models (cross-eval loss/ppl JSON metrics)")
    parser.add_argument("--baseline", action="store_true",
                        help="Compute baseline losses (base model)")
    parser.add_argument("--saves_dir", type=str, default=None,
                        help="Dir with unlearned model checkpoints")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--baseline_file", type=str, default=None,
                        help="Path to baseline cache JSON (default: extract-label/wikitext_baseline.json)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="WikiText data dir (default: auto-detect)")
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Optional shared test JSON evaluated for every model in addition to each triplet's own test.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: extract-label/wikitext_cross_metrics.json)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help='JSON field for text content (default: "text", use "qa_prompt" for QA prompts)',
    )
    parser.add_argument(
        "--triplets",
        type=str,
        default=None,
        help='Shorthand for setting both --eval_triplets and --model_triplets to the same triplet list.',
    )
    parser.add_argument(
        "--eval_triplets",
        type=str,
        default=None,
        help='Specific eval/data triplets, e.g. "triplet_001 triplet_021". Overrides --start/--end for the eval axis.',
    )
    parser.add_argument(
        "--model_triplets",
        type=str,
        default=None,
        help='Specific unlearn model triplets, e.g. "triplet_001 triplet_021". Overrides --start/--end for the model axis.',
    )
    parser.add_argument("--start", type=int, default=1,
                        help="First triplet index for range-based selection (default: 1)")
    parser.add_argument("--end", type=int, default=None,
                        help="Last triplet index for range-based selection (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a partial run")
    args = parser.parse_args()
    args.triplet_list = parse_triplet_selection(args.triplets)
    args.eval_triplet_list = parse_triplet_selection(args.eval_triplets)
    args.model_triplet_list = parse_triplet_selection(args.model_triplets)
    if args.eval_triplet_list is None and args.triplet_list is not None:
        args.eval_triplet_list = list(args.triplet_list)
    if args.model_triplet_list is None and args.triplet_list is not None:
        args.model_triplet_list = list(args.triplet_list)

    if args.baseline:
        compute_baselines(args)
    elif args.saves_dir:
        compute_labels(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
