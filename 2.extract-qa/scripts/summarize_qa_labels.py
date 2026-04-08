"""
Summarize QA label results: per-triplet and aggregate statistics comparing
base vs unlearned model correctness.

Reads pair files produced by eval_wikitext_qa.py and outputs:
  1. Per-triplet summary table  (CSV + console)
  2. Per-question detail        (CSV)
  3. Aggregate statistics        (JSON + console)
"""

import json
from pathlib import Path
import argparse

import pandas as pd

LABEL_DIR = Path(__file__).resolve().parent.parent
PAIRS_DIR = LABEL_DIR / "wikitext_qa_labels" / "pairs"


def load_pair(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_triplet_row(pair):
    c = pair["counts"]
    base_correct = c["base_correct_unlearn_correct"] + c["base_correct_unlearn_wrong"]
    base_wrong = c["base_wrong_unlearn_correct"] + c["base_wrong_unlearn_wrong"]
    forget_rate = c["base_correct_unlearn_wrong"] / base_correct if base_correct else 0.0
    gain_rate = c["base_wrong_unlearn_correct"] / base_wrong if base_wrong else 0.0

    return {
        "triplet": pair["eval_triplet"],
        "domain": pair.get("eval_domain", ""),
        "n_questions": pair["num_records"],
        "base_acc": pair["base_accuracy"],
        "unlearn_acc": pair["unlearn_accuracy"],
        "delta_acc": round(pair["unlearn_accuracy"] - pair["base_accuracy"], 4),
        "CC": c["base_correct_unlearn_correct"],
        "CW": c["base_correct_unlearn_wrong"],
        "WC": c["base_wrong_unlearn_correct"],
        "WW": c["base_wrong_unlearn_wrong"],
        "corrupt": c.get("corrupt", 0),
        "forget_rate": round(forget_rate, 4),
        "gain_rate": round(gain_rate, 4),
    }


def build_question_rows(pair):
    rows = []
    for ex in pair["examples"]:
        rows.append({
            "triplet": pair["eval_triplet"],
            "domain": pair.get("eval_domain", ""),
            "record_index": ex["record_index"],
            "question": ex["question"],
            "answer": ex["answer"],
            "base_prediction": ex["base_prediction"],
            "unlearn_prediction": ex["unlearn_prediction"],
            "base_correct": ex["base_correct"],
            "unlearn_correct": ex["unlearn_correct"],
            "case": ex["case"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize QA label results")
    parser.add_argument("--pairs_dir", type=str, default=str(PAIRS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(LABEL_DIR))
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_files = sorted(pairs_dir.glob("*.json"))
    if not pair_files:
        print(f"No pair files found in {pairs_dir}")
        return

    triplet_rows = []
    question_rows = []
    for pf in pair_files:
        pair = load_pair(pf)
        triplet_rows.append(build_triplet_row(pair))
        question_rows.extend(build_question_rows(pair))

    df_triplet = pd.DataFrame(triplet_rows)
    df_question = pd.DataFrame(question_rows)

    # ── Per-triplet summary ──────────────────────────────────────────────────
    triplet_csv = output_dir / "qa_summary_triplet.csv"
    df_triplet.to_csv(triplet_csv, index=False)

    # ── Per-question detail ──────────────────────────────────────────────────
    question_csv = output_dir / "qa_summary_questions.csv"
    df_question.to_csv(question_csv, index=False)

    # ── Aggregate stats ──────────────────────────────────────────────────────
    total_q = int(df_triplet["n_questions"].sum())
    total_cc = int(df_triplet["CC"].sum())
    total_cw = int(df_triplet["CW"].sum())
    total_wc = int(df_triplet["WC"].sum())
    total_ww = int(df_triplet["WW"].sum())
    total_corrupt = int(df_triplet["corrupt"].sum())
    total_base_correct = total_cc + total_cw
    total_unlearn_correct = total_cc + total_wc

    agg = {
        "n_pairs": len(df_triplet),
        "n_questions": total_q,
        "base_accuracy": round(total_base_correct / total_q, 4) if total_q else 0,
        "unlearn_accuracy": round(total_unlearn_correct / total_q, 4) if total_q else 0,
        "delta_accuracy": round((total_unlearn_correct - total_base_correct) / total_q, 4) if total_q else 0,
        "case_counts": {
            "CC (both_correct)": total_cc,
            "CW (base_correct→unlearn_wrong, forgotten)": total_cw,
            "WC (base_wrong→unlearn_correct, gained)": total_wc,
            "WW (both_wrong)": total_ww,
            "corrupt": total_corrupt,
        },
        "forget_rate (CW / base_correct)": round(total_cw / total_base_correct, 4) if total_base_correct else 0,
        "retain_rate (CC / base_correct)": round(total_cc / total_base_correct, 4) if total_base_correct else 0,
        "gain_rate (WC / base_wrong)": round(total_wc / (total_wc + total_ww), 4) if (total_wc + total_ww) else 0,
        "per_triplet_mean": {
            "base_acc": round(df_triplet["base_acc"].mean(), 4),
            "unlearn_acc": round(df_triplet["unlearn_acc"].mean(), 4),
            "delta_acc": round(df_triplet["delta_acc"].mean(), 4),
            "forget_rate": round(df_triplet["forget_rate"].mean(), 4),
        },
    }

    agg_json = output_dir / "qa_summary_aggregate.json"
    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    # ── Console output ───────────────────────────────────────────────────────
    print("=" * 90)
    print("  QA Label Summary: Base vs Unlearned")
    print("=" * 90)

    display_cols = ["triplet", "domain", "n_questions",
                    "base_acc", "unlearn_acc", "delta_acc",
                    "CC", "CW", "WC", "WW", "forget_rate"]
    print(df_triplet[display_cols].to_string(index=False))

    print(f"\n{'─' * 90}")
    print(f"  Pairs: {agg['n_pairs']}   Questions: {agg['n_questions']}")
    print(f"  Base accuracy (micro):    {agg['base_accuracy']:.4f}")
    print(f"  Unlearn accuracy (micro): {agg['unlearn_accuracy']:.4f}")
    print(f"  Delta accuracy:           {agg['delta_accuracy']:+.4f}")
    print(f"{'─' * 90}")
    print(f"  CC (both correct):   {total_cc:4d}  │  Retain rate: {agg['retain_rate (CC / base_correct)']:.4f}")
    print(f"  CW (forgotten):      {total_cw:4d}  │  Forget rate: {agg['forget_rate (CW / base_correct)']:.4f}")
    print(f"  WC (gained):         {total_wc:4d}  │  Gain rate:   {agg['gain_rate (WC / base_wrong)']:.4f}")
    print(f"  WW (both wrong):     {total_ww:4d}  │")
    print(f"  Corrupt:             {total_corrupt:4d}  │")
    print(f"{'─' * 90}")

    print(f"\n  Saved:")
    print(f"    {triplet_csv}")
    print(f"    {question_csv}")
    print(f"    {agg_json}")
    print("=" * 90)


if __name__ == "__main__":
    main()
