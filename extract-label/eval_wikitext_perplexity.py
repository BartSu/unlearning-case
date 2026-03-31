"""
Extract Y labels for WikiText unlearning experiments.

Labels computed per triplet (all are loss deltas = unlearn - base):
  Y_general:   loss delta on S_test   -- general knowledge corruption
  Y_forget:    loss delta on S_forget -- forget effectiveness (higher = forgot more)
  Y_retain:    loss delta on S_retain -- collateral damage (higher = more side damage)
  Y_precision: Y_forget / Y_general   -- unlearning precision (higher = more targeted)

Pipeline:
  1. Compute baselines (base model on test + each triplet's forget/retain):
     python eval_wikitext_perplexity.py --baseline --end 50 --resume

  2. Compute Y for all unlearned models:
     python eval_wikitext_perplexity.py --saves_dir ../data-preparation/open-unlearning/saves/unlearn --end 50 --resume

  Use --start / --end to limit triplet range.
  Both steps support --resume to continue from a partial run.
"""

import json
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


WIKITEXT_DIR = Path(__file__).resolve().parent.parent / "data-preparation" / "data" / "wikitext"
OUT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BASELINE_FILE = "wikitext_baseline.json"


def load_texts(json_path):
    with open(json_path) as f:
        return [item["text"] for item in json.load(f)]


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

        outputs = model(**encodings, labels=encodings["input_ids"])

        n_tokens = encodings["attention_mask"].sum().item()
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


def get_triplet_dirs(data_dir, start=1, end=None):
    dirs = sorted(data_dir.glob("triplet_*"))
    return [d for d in dirs
            if start <= int(d.name.split("_")[1]) <= (end or 9999)]


def _save_baseline(baseline, path):
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)


def extract_triplet_id(dirname):
    """Extract 'triplet_NNN' from a model directory name."""
    for part in dirname.split("_"):
        if part.startswith("triplet"):
            idx = dirname.index("triplet")
            return dirname[idx : idx + 11]
    return dirname


# ── Phase 1: Baselines ──────────────────────────────────────────────────────

def compute_baselines(args):
    """Base model loss on test set + each triplet's forget/retain sets."""
    baseline_path = OUT_DIR / BASELINE_FILE
    data_dir = Path(args.data_dir) if args.data_dir else WIKITEXT_DIR

    baseline = {}
    if args.resume and baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        n_cached = len(baseline.get("triplets", {}))
        print(f"Resuming from existing baseline ({n_cached} triplets cached)")

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = load_model(args.base_model)

    if "test" not in baseline:
        test_texts = load_texts(args.test_path or (data_dir / "test.json"))
        print(f"\nBaseline on test set ({len(test_texts)} texts) ...")
        loss, ppl = compute_avg_loss(model, tokenizer, test_texts,
                                     args.max_length, args.batch_size)
        baseline["model"] = args.base_model
        baseline["test"] = {"loss": round(loss, 6), "ppl": round(ppl, 2)}
        print(f"  test: loss={loss:.6f}  ppl={ppl:.2f}")
        _save_baseline(baseline, baseline_path)

    if "triplets" not in baseline:
        baseline["triplets"] = {}

    triplet_dirs = get_triplet_dirs(data_dir, args.start, args.end)
    done = set(baseline["triplets"].keys())
    remaining = [d for d in triplet_dirs if d.name not in done]
    print(f"\nPer-triplet baselines: {len(remaining)} remaining "
          f"(of {len(triplet_dirs)} total, range {args.start}-{args.end or 'all'})")

    for i, tdir in enumerate(remaining, 1):
        name = tdir.name
        forget_texts = load_texts(tdir / "train.json")
        retain_texts = load_texts(tdir / "val.json")

        f_loss, f_ppl = compute_avg_loss(model, tokenizer, forget_texts,
                                         args.max_length, args.batch_size)
        r_loss, r_ppl = compute_avg_loss(model, tokenizer, retain_texts,
                                         args.max_length, args.batch_size)

        baseline["triplets"][name] = {
            "forget": {"loss": round(f_loss, 6), "ppl": round(f_ppl, 2)},
            "retain": {"loss": round(r_loss, 6), "ppl": round(r_ppl, 2)},
        }
        print(f"  [{i:3d}/{len(remaining)}] {name}: "
              f"forget={f_loss:.4f}  retain={r_loss:.4f}")

        if i % 5 == 0 or i == len(remaining):
            _save_baseline(baseline, baseline_path)

    _save_baseline(baseline, baseline_path)
    print(f"\nBaseline saved to {baseline_path}")

    del model
    torch.cuda.empty_cache()
    return baseline


# ── Phase 2: Y labels ───────────────────────────────────────────────────────

def compute_labels(args):
    """Compute all Y labels for unlearned models."""
    baseline_path = OUT_DIR / BASELINE_FILE
    data_dir = Path(args.data_dir) if args.data_dir else WIKITEXT_DIR
    saves = Path(args.saves_dir)
    output_path = Path(args.output) if args.output else OUT_DIR / "wikitext_labels.csv"

    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Loaded baselines ({len(baseline.get('triplets', {}))} triplets)")
    else:
        print("No baseline cache found, computing baselines first ...")
        baseline = compute_baselines(args)

    base_test = baseline["test"]
    print(f"Base test: loss={base_test['loss']:.6f}  ppl={base_test['ppl']:.2f}")

    model_dirs = sorted(saves.glob("wikitext_*_triplet_*_GradAscent"))
    if not model_dirs:
        model_dirs = sorted(saves.glob("*triplet*"))

    if args.start or args.end:
        lo, hi = args.start, args.end or 9999
        def _in_range(d):
            tid = extract_triplet_id(d.name)
            try:
                return lo <= int(tid.split("_")[1]) <= hi
            except (ValueError, IndexError):
                return False
        model_dirs = [d for d in model_dirs if _in_range(d)]

    print(f"Found {len(model_dirs)} unlearned models "
          f"(range {args.start}-{args.end or 'all'})\n")

    rows = []
    done_splits = set()
    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(output_path)
            rows = existing.to_dict("records")
            done_splits = set(existing["split"])
            print(f"Resuming: {len(done_splits)} triplets already evaluated")
        except pd.errors.EmptyDataError:
            print("Existing output file is empty, starting fresh")

    test_texts = load_texts(args.test_path or (data_dir / "test.json"))

    for mdir in tqdm(model_dirs, desc="Evaluating"):
        triplet_id = extract_triplet_id(mdir.name)
        if triplet_id in done_splits:
            continue

        tdir = data_dir / triplet_id
        if not tdir.exists():
            print(f"  SKIP {triplet_id}: data dir not found")
            continue

        triplet_base = baseline.get("triplets", {}).get(triplet_id)
        if not triplet_base:
            print(f"  SKIP {triplet_id}: no baseline cached")
            continue

        forget_texts = load_texts(tdir / "train.json")
        retain_texts = load_texts(tdir / "val.json")

        model, tokenizer = load_model(str(mdir))

        test_loss, test_ppl = compute_avg_loss(
            model, tokenizer, test_texts, args.max_length, args.batch_size)
        forget_loss, forget_ppl = compute_avg_loss(
            model, tokenizer, forget_texts, args.max_length, args.batch_size)
        retain_loss, retain_ppl = compute_avg_loss(
            model, tokenizer, retain_texts, args.max_length, args.batch_size)

        y_general = test_loss - base_test["loss"]
        y_forget = forget_loss - triplet_base["forget"]["loss"]
        y_retain = retain_loss - triplet_base["retain"]["loss"]
        y_precision = (y_forget / y_general) if abs(y_general) > 1e-8 else 0.0

        rows.append({
            "split": triplet_id,
            # S_test (general knowledge corruption)
            "base_test_loss": base_test["loss"],
            "unlearn_test_loss": round(test_loss, 6),
            "base_test_ppl": base_test["ppl"],
            "unlearn_test_ppl": round(test_ppl, 2),
            "Y_general": round(y_general, 6),
            # S_forget (forget effectiveness)
            "base_forget_loss": triplet_base["forget"]["loss"],
            "unlearn_forget_loss": round(forget_loss, 6),
            "base_forget_ppl": triplet_base["forget"]["ppl"],
            "unlearn_forget_ppl": round(forget_ppl, 2),
            "Y_forget": round(y_forget, 6),
            # S_retain (collateral damage)
            "base_retain_loss": triplet_base["retain"]["loss"],
            "unlearn_retain_loss": round(retain_loss, 6),
            "base_retain_ppl": triplet_base["retain"]["ppl"],
            "unlearn_retain_ppl": round(retain_ppl, 2),
            "Y_retain": round(y_retain, 6),
            # Derived
            "Y_precision": round(y_precision, 6),
        })

        print(f"  {triplet_id}: Y_general={y_general:+.4f}  "
              f"Y_forget={y_forget:+.4f}  Y_retain={y_retain:+.4f}  "
              f"precision={y_precision:.2f}")

        del model
        torch.cuda.empty_cache()

        pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"\nLabels saved to {output_path}  ({len(df)} rows)")
        summary_cols = ["split", "Y_general", "Y_forget", "Y_retain", "Y_precision"]
        print(f"\n{df[summary_cols].to_string(index=False)}")
    else:
        print("\nNo models evaluated. Check --saves_dir path and triplet range.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract Y labels for WikiText unlearning")
    parser.add_argument("--baseline", action="store_true",
                        help="Compute baseline losses (base model)")
    parser.add_argument("--saves_dir", type=str, default=None,
                        help="Dir with unlearned model checkpoints")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="WikiText data dir (default: auto-detect)")
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--start", type=int, default=1,
                        help="First triplet index (default: 1)")
    parser.add_argument("--end", type=int, default=None,
                        help="Last triplet index (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a partial run")
    args = parser.parse_args()

    if args.baseline:
        compute_baselines(args)
    elif args.saves_dir:
        compute_labels(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
