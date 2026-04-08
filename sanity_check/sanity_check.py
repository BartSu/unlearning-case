import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_models
import math
import json
import time

ORIGINAL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

QA_PAIRS = {
    "forget": {
        "question": "What does Hsiao Yun-Hwa identify as in terms of gender?",
        "answer": "Hsiao Yun-Hwa is part of the LGBTQ+ community.",
    },
    "retain": {
        "question": "Are the details of Jaime Vasquez's birth documented?",
        "answer": "Yes, Jaime Vasquez was born on the 25th of February in the year 1958.",
    },
}


def compute_ppl(model, tokenizer, question, answer):
    q_ids = tokenizer(question, add_special_tokens=True, return_tensors="pt")["input_ids"]
    full_text = question + " " + answer
    full_ids = tokenizer(full_text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(model.device)

    labels = full_ids.clone()
    labels[:, : q_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(input_ids=full_ids, labels=labels)
    return math.exp(outputs.loss.item())


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def get_graddiff_models():
    models = list_models(author="open-unlearning", search="tofu_Llama-3.2-1B-Instruct_forget10_GradDiff")
    return sorted([m.id for m in models if "GradDiff" in m.id])


def main():
    unlearn_models = get_graddiff_models()
    print(f"Found {len(unlearn_models)} GradDiff models\n")

    print("Loading original model:", ORIGINAL_MODEL)
    orig_model, orig_tok = load_model(ORIGINAL_MODEL)
    orig_ppls = {n: compute_ppl(orig_model, orig_tok, p["question"], p["answer"]) for n, p in QA_PAIRS.items()}
    del orig_model
    torch.cuda.empty_cache()
    print(f"  Original PPL -> forget: {orig_ppls['forget']:.4f}, retain: {orig_ppls['retain']:.4f}\n")

    results = []
    for i, model_id in enumerate(unlearn_models):
        short_name = model_id.split("forget10_")[-1]
        t0 = time.time()
        print(f"[{i+1}/{len(unlearn_models)}] {short_name} ... ", end="", flush=True)
        try:
            model, tok = load_model(model_id)
            ppls = {n: compute_ppl(model, tok, p["question"], p["answer"]) for n, p in QA_PAIRS.items()}
            del model
            torch.cuda.empty_cache()

            row = {
                "model": model_id,
                "short": short_name,
                "forget_ppl": ppls["forget"],
                "retain_ppl": ppls["retain"],
                "forget_ratio": ppls["forget"] / orig_ppls["forget"],
                "retain_ratio": ppls["retain"] / orig_ppls["retain"],
            }
            results.append(row)
            print(f"forget={ppls['forget']:.4f} ({row['forget_ratio']:.2f}x)  "
                  f"retain={ppls['retain']:.4f} ({row['retain_ratio']:.2f}x)  "
                  f"[{time.time()-t0:.1f}s]")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"model": model_id, "short": short_name, "error": str(e)})

    print("\n" + "=" * 100)
    print(f"{'Model':<55} {'Forget PPL':>11} {'(ratio)':>8} {'Retain PPL':>12} {'(ratio)':>8}")
    print("-" * 100)
    print(f"{'ORIGINAL (baseline)':<55} {orig_ppls['forget']:>11.4f} {'1.00x':>8} {orig_ppls['retain']:>12.4f} {'1.00x':>8}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['short']:<55} {'ERROR':>11}")
            continue
        print(f"{r['short']:<55} {r['forget_ppl']:>11.4f} {r['forget_ratio']:>7.2f}x {r['retain_ppl']:>12.4f} {r['retain_ratio']:>7.2f}x")
    print("=" * 100)

    out_path = "sanity_check_graddiff_results.json"
    with open(out_path, "w") as f:
        json.dump({"original_ppls": orig_ppls, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
