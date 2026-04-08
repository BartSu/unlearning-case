"""
python scripts/eval_wikitext_qa.py --baseline --resume --triplets "triplet_001 triplet_010 triplet_050"

python scripts/eval_wikitext_qa.py \
  --saves_dir ../data-preparation/unlearn/saves/wikitext_unlearn \
  --resume --triplets "triplet_001 triplet_010 triplet_050"

Evaluate WikiText QA triplets by answer correctness.

Output artifacts:
  - baseline JSON: base-model predictions/correctness per QA record, grouped by
    triplet
  - label manifest + per-pair JSON files: base vs unlearn predictions and
    corrupt/normal labels for each QA record

Pipeline:
  1. Compute baseline predictions:
     python eval_wikitext_qa.py --baseline --resume

  2. Compare unlearn models against the baseline:
     python eval_wikitext_qa.py --saves_dir ../data-preparation/unlearn/saves/wikitext_unlearn --resume

By default, pair mode is "aligned": each unlearn model is evaluated only on the
matching QA triplet. Use --pair_mode cross to evaluate every
(unlearn model triplet, eval triplet) pair.

Label rule:
  - corrupt: base correct and unlearn wrong
  - normal: every other case
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QA_DIR_CANDIDATES = (
    PROJECT_ROOT / "data-preparation" / "data" / "wikitext_hdbscan_triplets_qa",
    PROJECT_ROOT / "data-preparation" / "data" / "wikitext_dbscan_triplets_qa",
)
OUT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_INPUT_FILENAME = "train.json"
BASELINE_FILE = "wikitext_qa_baseline.json"
DEFAULT_OUTPUT_DIR = "wikitext_qa_labels"
MANIFEST_FILE = "manifest.json"
PAIR_SUBDIR = "pairs"
BASELINE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
PAIR_SCHEMA_VERSION = 1
CASE_KEYS = (
    "base_correct_unlearn_wrong",
    "base_correct_unlearn_correct",
    "base_wrong_unlearn_wrong",
    "base_wrong_unlearn_correct",
)
LABEL_KEYS = ("corrupt", "normal")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_model(model_path, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
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


def resolve_baseline_path(args):
    if args.baseline_file:
        return Path(args.baseline_file)
    return OUT_DIR / BASELINE_FILE


def resolve_output_dir(args):
    if args.output_dir:
        return Path(args.output_dir)
    return OUT_DIR / DEFAULT_OUTPUT_DIR


def resolve_data_dir(data_dir_arg):
    if data_dir_arg:
        data_dir = Path(data_dir_arg)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"WikiText QA data dir not found: {data_dir}")
        return data_dir

    for candidate in QA_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate

    tried = ", ".join(str(path) for path in QA_DIR_CANDIDATES)
    raise FileNotFoundError(
        f"Could not auto-detect WikiText QA triplet dir. Tried: {tried}. "
        "Pass --data_dir explicitly."
    )


def resolve_triplet_qa_path(triplet_dir, input_filename):
    path = triplet_dir / input_filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {input_filename} under {triplet_dir}")
    return path


def load_qa_records(json_path, limit_per_triplet=None):
    payload = load_json(json_path)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list in {json_path}")

    if limit_per_triplet is not None:
        payload = payload[:limit_per_triplet]

    required_fields = ("question", "answer")
    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise RuntimeError(f"Expected item {idx} in {json_path} to be a JSON object")
        for field_name in required_fields:
            if not isinstance(record.get(field_name), str):
                raise RuntimeError(
                    f"Expected item {idx} in {json_path} to have a string "
                    f"'{field_name}' field"
                )

    return payload


def extract_triplet_meta(records):
    meta = {}
    for key in ("cluster_label", "domain", "domain_triplet_index"):
        for record in records:
            value = record.get(key)
            if value is not None:
                meta[key] = value
                break
    return meta


def normalize_text(text):
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "@-@": "-",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    normalized = normalized.lower()
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"[^0-9a-z\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def token_f1(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def answer_is_correct(prediction, answer):
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(answer)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()

    if pred_norm in gold_norm or gold_norm in pred_norm:
        pred_len = len(pred_tokens)
        gold_len = len(gold_tokens)
        shorter_len = min(pred_len, gold_len)
        longer_len = max(pred_len, gold_len)
        shorter_text = pred_norm if pred_len <= gold_len else gold_norm
        has_digit = any(ch.isdigit() for ch in shorter_text)

        if gold_len == 1 and gold_norm in pred_norm:
            return True
        if has_digit:
            return True
        if longer_len > 0 and shorter_len / longer_len >= 0.5:
            return True

    return token_f1(pred_tokens, gold_tokens) >= 0.8


def build_eval_prompt(question: str) -> str:
    """Closed-book prompt: question only, no passage."""
    return f"Question:\n{question}\n\nAnswer:"


def format_prompt(tokenizer, prompt_text):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError, RuntimeError):
            pass
    return prompt_text


def extract_answer_text(text):
    cleaned = str(text or "")
    cleaned = cleaned.replace("<|eot_id|>", " ").replace("</s>", " ").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        if lines[0].lower() == "assistant" and len(lines) > 1:
            cleaned = lines[1]
        else:
            cleaned = lines[0]

    match = re.match(r"^(?:assistant|answer)\s*[:\-]?\s*(.*)$", cleaned, flags=re.IGNORECASE)
    if match and match.group(1).strip():
        cleaned = match.group(1).strip()

    for marker in ("Question:", "Passage:", "Answer the question"):
        marker_idx = cleaned.find(marker)
        if marker_idx > 0:
            cleaned = cleaned[:marker_idx].strip()

    return cleaned.strip(" \t\n\r\"'")


def get_terminator_ids(tokenizer):
    terminators = []
    for token_id in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in terminators:
            terminators.append(token_id)
    return terminators


@torch.no_grad()
def generate_predictions(model, tokenizer, records, max_length, max_new_tokens, batch_size):
    model.eval()
    device = next(model.parameters()).device
    prompts = [format_prompt(tokenizer, build_eval_prompt(record["question"])) for record in records]
    predictions = []
    terminators = get_terminator_ids(tokenizer)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if terminators:
        generation_kwargs["eos_token_id"] = terminators if len(terminators) > 1 else terminators[0]

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        outputs = model.generate(**encodings, **generation_kwargs)
        generated_tokens = outputs[:, encodings["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions.extend(extract_answer_text(text) for text in decoded)

    return predictions


def accuracy(num_correct, num_records):
    if num_records <= 0:
        return 0.0
    return round(num_correct / num_records, 6)


def build_baseline_entry(records, predictions):
    output_records = []
    num_correct = 0
    for idx, (record, prediction) in enumerate(zip(records, predictions)):
        correct = answer_is_correct(prediction, record["answer"])
        num_correct += int(correct)
        output_records.append(
            {
                "record_index": idx,
                "source_train_index": record.get("source_train_index"),
                "prediction": prediction,
                "correct": correct,
            }
        )

    entry = extract_triplet_meta(records)
    entry.update(
        {
            "num_records": len(records),
            "num_correct": num_correct,
            "accuracy": accuracy(num_correct, len(records)),
            "records": output_records,
        }
    )
    return entry


def has_complete_baseline_entry(entry, expected_num_records):
    if not isinstance(entry, dict):
        return False
    records = entry.get("records")
    if entry.get("num_records") != expected_num_records or not isinstance(records, list):
        return False
    if len(records) != expected_num_records:
        return False

    for record in records:
        if not isinstance(record, dict):
            return False
        if not isinstance(record.get("record_index"), int):
            return False
        if not isinstance(record.get("prediction"), str):
            return False
        if not isinstance(record.get("correct"), bool):
            return False

    return True


def classify_label(base_correct, unlearn_correct):
    if base_correct and not unlearn_correct:
        return "corrupt", "base_correct_unlearn_wrong"
    if base_correct and unlearn_correct:
        return "normal", "base_correct_unlearn_correct"
    if (not base_correct) and (not unlearn_correct):
        return "normal", "base_wrong_unlearn_wrong"
    return "normal", "base_wrong_unlearn_correct"


def build_pair_payload(
    *,
    model_triplet,
    model_dir,
    eval_triplet,
    qa_path,
    records,
    baseline_entry,
    unlearn_predictions,
    base_model,
    pair_mode,
):
    base_records = baseline_entry["records"]
    if len(base_records) != len(records):
        raise RuntimeError(
            f"Baseline length mismatch for {eval_triplet}: "
            f"{len(base_records)} cached vs {len(records)} current records"
        )

    examples = []
    counts = {key: 0 for key in LABEL_KEYS + CASE_KEYS}
    base_num_correct = 0
    unlearn_num_correct = 0
    eval_meta = extract_triplet_meta(records)

    for idx, (record, base_record, unlearn_prediction) in enumerate(
        zip(records, base_records, unlearn_predictions)
    ):
        base_prediction = base_record["prediction"]
        base_correct = bool(base_record["correct"])
        unlearn_correct = answer_is_correct(unlearn_prediction, record["answer"])
        label, case_key = classify_label(base_correct, unlearn_correct)

        counts[label] += 1
        counts[case_key] += 1
        base_num_correct += int(base_correct)
        unlearn_num_correct += int(unlearn_correct)
        examples.append(
            {
                "record_index": idx,
                "source_train_index": record.get("source_train_index"),
                "question": record["question"],
                "answer": record["answer"],
                "base_prediction": base_prediction,
                "unlearn_prediction": unlearn_prediction,
                "base_correct": base_correct,
                "unlearn_correct": unlearn_correct,
                "label": label,
                "case": case_key,
            }
        )

    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "pair_mode": pair_mode,
        "base_model": base_model,
        "model_triplet": model_triplet,
        "model_dir": str(model_dir),
        "eval_triplet": eval_triplet,
        "data_path": str(qa_path),
        "input_filename": qa_path.name,
        "eval_domain": eval_meta.get("domain"),
        "eval_cluster_label": eval_meta.get("cluster_label"),
        "eval_domain_triplet_index": eval_meta.get("domain_triplet_index"),
        "num_records": len(records),
        "counts": counts,
        "base_num_correct": base_num_correct,
        "base_accuracy": accuracy(base_num_correct, len(records)),
        "unlearn_num_correct": unlearn_num_correct,
        "unlearn_accuracy": accuracy(unlearn_num_correct, len(records)),
        "examples": examples,
    }


def has_complete_pair_payload(payload, expected_num_records):
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != PAIR_SCHEMA_VERSION:
        return False
    if payload.get("num_records") != expected_num_records:
        return False

    counts = payload.get("counts")
    examples = payload.get("examples")
    if not isinstance(counts, dict) or not isinstance(examples, list):
        return False
    if len(examples) != expected_num_records:
        return False

    for label_key in LABEL_KEYS:
        if not isinstance(counts.get(label_key), int):
            return False
    for case_key in CASE_KEYS:
        if not isinstance(counts.get(case_key), int):
            return False
    return True


def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def build_pair_file_path(output_dir, model_dir, eval_triplet):
    filename = f"{sanitize_filename(model_dir.name)}__{eval_triplet}.json"
    return output_dir / PAIR_SUBDIR / filename


def build_manifest_entry(pair_payload, output_dir, pair_file):
    return {
        "model_triplet": pair_payload["model_triplet"],
        "model_dir": pair_payload["model_dir"],
        "eval_triplet": pair_payload["eval_triplet"],
        "eval_domain": pair_payload.get("eval_domain"),
        "eval_cluster_label": pair_payload.get("eval_cluster_label"),
        "eval_domain_triplet_index": pair_payload.get("eval_domain_triplet_index"),
        "pair_file": pair_file.relative_to(output_dir).as_posix(),
        "num_records": pair_payload["num_records"],
        "counts": dict(pair_payload["counts"]),
        "base_num_correct": pair_payload["base_num_correct"],
        "base_accuracy": pair_payload["base_accuracy"],
        "unlearn_num_correct": pair_payload["unlearn_num_correct"],
        "unlearn_accuracy": pair_payload["unlearn_accuracy"],
    }


def sorted_manifest_pairs(pair_entries):
    return sorted(
        pair_entries.values(),
        key=lambda row: (
            str(row.get("model_triplet")),
            str(row.get("eval_triplet")),
            str(row.get("model_dir")),
        ),
    )


def extract_triplet_id(dirname):
    return canonicalize_triplet_name(dirname)


def resolve_model_dirs(saves, start, end, model_triplets):
    if (saves / "config.json").exists():
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
    elif start or end:
        lo, hi = start, end or 9999

        def _in_range(model_dir):
            triplet_id = extract_triplet_id(model_dir.name)
            try:
                return lo <= int(triplet_id.split("_")[1]) <= hi
            except (ValueError, IndexError):
                return False

        model_dirs = [d for d in model_dirs if _in_range(d)]

    return model_dirs


def build_pair_plan(model_dirs, eval_triplet_dirs, pair_mode):
    eval_lookup = {triplet_dir.name: triplet_dir for triplet_dir in eval_triplet_dirs}
    pair_plan = []
    missing = []

    for model_dir in model_dirs:
        model_triplet = extract_triplet_id(model_dir.name)
        if pair_mode == "aligned":
            eval_dir = eval_lookup.get(model_triplet)
            if eval_dir is None:
                missing.append((model_dir, model_triplet))
                continue
            pair_plan.append((model_dir, model_triplet, eval_dir))
            continue

        for eval_dir in eval_triplet_dirs:
            pair_plan.append((model_dir, model_triplet, eval_dir))

    return pair_plan, missing


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

    cached_input_filename = baseline.get("input_filename")
    if cached_input_filename and cached_input_filename != args.input_filename:
        raise ValueError(
            f"Baseline cache was created for {cached_input_filename}, not {args.input_filename}. "
            "Use a different --baseline_file or delete the stale cache."
        )


def compute_baselines(args):
    baseline_path = resolve_baseline_path(args)
    data_dir = resolve_data_dir(args.data_dir)
    selected_triplets = getattr(args, "eval_triplet_list", None)
    triplet_dirs = get_triplet_dirs(
        data_dir,
        start=args.start,
        end=args.end,
        selected_triplets=selected_triplets,
    )
    if not triplet_dirs:
        print("\nNo QA triplets found. Check --data_dir and triplet selection.")
        return {}

    baseline = {}
    if args.resume and baseline_path.exists():
        baseline = load_json(baseline_path)
        baseline.setdefault("triplets", {})
        n_cached = 0
        for triplet_dir in triplet_dirs:
            qa_path = resolve_triplet_qa_path(triplet_dir, args.input_filename)
            records = load_qa_records(qa_path, limit_per_triplet=args.limit_per_triplet)
            if has_complete_baseline_entry(baseline["triplets"].get(triplet_dir.name), len(records)):
                n_cached += 1
        print(f"Resuming from existing baseline ({n_cached} triplets cached)")

    ensure_baseline_compatible(baseline, args, data_dir)

    baseline["schema_version"] = BASELINE_SCHEMA_VERSION
    baseline["evaluation_scope"] = "qa_triplet"
    baseline["model"] = args.base_model
    baseline["data_dir"] = str(data_dir)
    baseline["input_filename"] = args.input_filename
    baseline["limit_per_triplet"] = args.limit_per_triplet
    baseline.setdefault("triplets", {})

    remaining = []
    for triplet_dir in triplet_dirs:
        qa_path = resolve_triplet_qa_path(triplet_dir, args.input_filename)
        records = load_qa_records(qa_path, limit_per_triplet=args.limit_per_triplet)
        if not has_complete_baseline_entry(baseline["triplets"].get(triplet_dir.name), len(records)):
            remaining.append(triplet_dir)

    selection_label = describe_triplet_selection(selected_triplets, args.start, args.end)
    print(
        f"\nPer-triplet QA baselines: {len(remaining)} remaining "
        f"(of {len(triplet_dirs)} total, {selection_label})"
    )

    if not remaining:
        save_json(baseline_path, baseline)
        print(f"\nBaseline already complete at {baseline_path}")
        return baseline

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = load_model(args.base_model)

    for i, triplet_dir in enumerate(remaining, 1):
        qa_path = resolve_triplet_qa_path(triplet_dir, args.input_filename)
        records = load_qa_records(qa_path, limit_per_triplet=args.limit_per_triplet)
        predictions = generate_predictions(
            model,
            tokenizer,
            records,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        baseline_entry = build_baseline_entry(records, predictions)
        baseline["triplets"][triplet_dir.name] = baseline_entry
        print(
            f"  [{i:3d}/{len(remaining)}] {triplet_dir.name}: "
            f"acc={baseline_entry['accuracy']:.4f}  "
            f"correct={baseline_entry['num_correct']}/{baseline_entry['num_records']}"
        )
        save_json(baseline_path, baseline)

    print(f"\nBaseline saved to {baseline_path}")
    del model
    torch.cuda.empty_cache()
    return baseline


def compute_labels(args):
    baseline_path = resolve_baseline_path(args)
    data_dir = resolve_data_dir(args.data_dir)
    saves = Path(args.saves_dir)
    output_dir = resolve_output_dir(args)
    manifest_path = output_dir / MANIFEST_FILE
    eval_triplets = getattr(args, "eval_triplet_list", None)
    model_triplets = getattr(args, "model_triplet_list", None)

    if baseline_path.exists():
        baseline = load_json(baseline_path)
        baseline.setdefault("triplets", {})
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

    missing_baselines = []
    for triplet_dir in eval_triplet_dirs:
        qa_path = resolve_triplet_qa_path(triplet_dir, args.input_filename)
        records = load_qa_records(qa_path, limit_per_triplet=args.limit_per_triplet)
        if not has_complete_baseline_entry(baseline.get("triplets", {}).get(triplet_dir.name), len(records)):
            missing_baselines.append(triplet_dir.name)
    if missing_baselines:
        print(
            f"Missing baselines for {len(missing_baselines)} triplets, computing them now ..."
        )
        baseline = compute_baselines(args)
        baseline.setdefault("triplets", {})

    model_dirs = resolve_model_dirs(
        saves,
        start=args.start,
        end=args.end,
        model_triplets=model_triplets,
    )
    if not model_dirs:
        print("\nNo unlearned models found. Check --saves_dir and triplet selection.")
        return

    pair_plan, missing_aligned = build_pair_plan(model_dirs, eval_triplet_dirs, args.pair_mode)
    if missing_aligned:
        missing_triplets = [triplet for _, triplet in missing_aligned]
        print(
            "Skipping aligned models without matching QA triplets: "
            f"{', '.join(missing_triplets)}"
        )
    if not pair_plan:
        print("\nNo model/eval pairs to run. Check --pair_mode and triplet selection.")
        return

    model_triplet_names = list(dict.fromkeys(extract_triplet_id(d.name) for d in model_dirs))
    eval_triplet_names = [triplet_dir.name for triplet_dir in eval_triplet_dirs]
    model_selection_label = describe_triplet_selection(model_triplets, args.start, args.end)
    eval_selection_label = describe_triplet_selection(eval_triplets, args.start, args.end)
    print(f"Found {len(model_dirs)} unlearned models ({model_selection_label})")
    print(f"Evaluating on {len(eval_triplet_dirs)} QA triplets ({eval_selection_label})")
    print(f"Pair mode: {args.pair_mode}")
    print(f"Total model/eval pairs: {len(pair_plan)}\n")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "evaluation_scope": "qa_triplet_labels",
        "pair_mode": args.pair_mode,
        "baseline_file": str(baseline_path),
        "base_model": baseline.get("model", args.base_model),
        "data_dir": str(data_dir),
        "input_filename": args.input_filename,
        "limit_per_triplet": args.limit_per_triplet,
        "saves_dir": str(saves),
        "output_dir": str(output_dir),
        "range": {"start": args.start, "end": args.end},
        "eval_triplets": eval_triplet_names,
        "model_triplets": model_triplet_names,
        "pairs": [],
    }

    pair_entries = {}
    planned_keys = {(str(model_dir), eval_dir.name) for model_dir, _, eval_dir in pair_plan}
    done_keys = set()
    if args.resume and manifest_path.exists() and manifest_path.stat().st_size > 0:
        try:
            existing = load_json(manifest_path)
            if (
                isinstance(existing, dict)
                and existing.get("schema_version") == MANIFEST_SCHEMA_VERSION
                and existing.get("evaluation_scope") == "qa_triplet_labels"
                and existing.get("pair_mode") == args.pair_mode
                and existing.get("baseline_file") == str(baseline_path)
                and existing.get("data_dir") == str(data_dir)
                and existing.get("input_filename") == args.input_filename
                and existing.get("saves_dir") == str(saves)
                and isinstance(existing.get("pairs"), list)
            ):
                manifest = existing
                for entry in manifest.get("pairs", []):
                    key = (entry.get("model_dir"), entry.get("eval_triplet"))
                    if not key[0] or not key[1]:
                        continue
                    pair_entries[key] = entry
                    if key not in planned_keys:
                        continue
                    pair_file = output_dir / str(entry.get("pair_file", ""))
                    try:
                        if not pair_file.is_file():
                            continue
                        pair_payload = load_json(pair_file)
                        if has_complete_pair_payload(pair_payload, entry.get("num_records")):
                            done_keys.add(key)
                    except (json.JSONDecodeError, OSError, RuntimeError):
                        continue
                print(f"Resuming: {len(done_keys)} model/eval pairs already evaluated")
            else:
                print("Existing output uses a different format, starting fresh")
        except json.JSONDecodeError:
            print("Existing output is not valid JSON, starting fresh")

    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["evaluation_scope"] = "qa_triplet_labels"
    manifest["pair_mode"] = args.pair_mode
    manifest["baseline_file"] = str(baseline_path)
    manifest["base_model"] = baseline.get("model", args.base_model)
    manifest["data_dir"] = str(data_dir)
    manifest["input_filename"] = args.input_filename
    manifest["limit_per_triplet"] = args.limit_per_triplet
    manifest["saves_dir"] = str(saves)
    manifest["output_dir"] = str(output_dir)
    manifest["range"] = {"start": args.start, "end": args.end}
    manifest["eval_triplets"] = eval_triplet_names
    manifest["model_triplets"] = model_triplet_names
    manifest["pairs"] = sorted_manifest_pairs(pair_entries)
    save_json(manifest_path, manifest)

    pairs_by_model: Dict[Path, List[Tuple[str, Path]]] = {}
    for model_dir, model_triplet, eval_dir in pair_plan:
        pairs_by_model.setdefault(model_dir, []).append((model_triplet, eval_dir))

    progress = tqdm(total=len(pair_plan), initial=len(done_keys), desc="Evaluating pairs")
    for model_dir in model_dirs:
        scheduled = pairs_by_model.get(model_dir, [])
        if not scheduled:
            continue

        if all((str(model_dir), eval_dir.name) in done_keys for _, eval_dir in scheduled):
            continue

        print(f"Loading unlearn model: {model_dir}")
        model, tokenizer = load_model(str(model_dir))
        new_pairs_for_model = 0

        for model_triplet, eval_dir in scheduled:
            pair_key = (str(model_dir), eval_dir.name)
            if pair_key in done_keys:
                continue

            qa_path = resolve_triplet_qa_path(eval_dir, args.input_filename)
            records = load_qa_records(qa_path, limit_per_triplet=args.limit_per_triplet)
            baseline_entry = baseline.get("triplets", {}).get(eval_dir.name)
            if not has_complete_baseline_entry(baseline_entry, len(records)):
                print(f"  SKIP {model_triplet} x {eval_dir.name}: no baseline cached")
                progress.update(1)
                continue

            predictions = generate_predictions(
                model,
                tokenizer,
                records,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
            )
            pair_payload = build_pair_payload(
                model_triplet=model_triplet,
                model_dir=model_dir,
                eval_triplet=eval_dir.name,
                qa_path=qa_path,
                records=records,
                baseline_entry=baseline_entry,
                unlearn_predictions=predictions,
                base_model=baseline.get("model", args.base_model),
                pair_mode=args.pair_mode,
            )
            pair_file = build_pair_file_path(output_dir, model_dir, eval_dir.name)
            save_json(pair_file, pair_payload)
            pair_entries[pair_key] = build_manifest_entry(pair_payload, output_dir, pair_file)
            done_keys.add(pair_key)
            manifest["pairs"] = sorted_manifest_pairs(pair_entries)
            save_json(manifest_path, manifest)
            new_pairs_for_model += 1
            progress.update(1)

        print(f"  {extract_triplet_id(model_dir.name)}: evaluated {new_pairs_for_model} new QA pairs")
        del model
        torch.cuda.empty_cache()

    progress.close()

    planned_entries = [pair_entries[key] for key in planned_keys if key in pair_entries]
    if planned_entries:
        print(f"\nLabels saved under {output_dir}  ({len(planned_entries)} pair files)")
    else:
        print("\nNo QA labels were written. Check --saves_dir, --pair_mode, and triplet selection.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WikiText QA triplets by answer correctness."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Compute base-model QA predictions/correctness only.",
    )
    parser.add_argument(
        "--saves_dir",
        type=str,
        default=None,
        help="Dir with unlearned model checkpoints.",
    )
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--baseline_file",
        type=str,
        default=None,
        help="Path to baseline cache JSON (default: extract-label/wikitext_qa_baseline.json).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="WikiText QA triplet dir (default: auto-detect).",
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default=DEFAULT_INPUT_FILENAME,
        help="Per-triplet QA filename to evaluate (default: train.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for QA label outputs (default: extract-label/wikitext_qa_labels).",
    )
    parser.add_argument(
        "--pair_mode",
        type=str,
        choices=("aligned", "cross"),
        default="aligned",
        help="aligned: only matching model/eval triplets, cross: cartesian evaluation.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--limit_per_triplet",
        type=int,
        default=None,
        help="Optional debug limit on QA records evaluated per triplet.",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial baseline or QA label outputs.",
    )
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
