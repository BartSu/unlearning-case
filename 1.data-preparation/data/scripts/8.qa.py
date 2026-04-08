"""
Generate QA-style train splits from triplet train.json files with a local vLLM model.

Default input:
  data/wikitext_hdbscan_triplets/
    triplet_001/train.json
    triplet_002/train.json
    ...
    run_manifest.json

Default output:
  data/wikitext_hdbscan_triplets_qa/
    triplet_001/train.json
    triplet_002/train.json
    ...
    run_manifest.json

For every source train record with a ``text`` field, the script loads a local
vLLM engine (default: ``openai/gpt-oss-20b``) and generates:
  - question: a grounded QA question answerable only from the text
  - answer: a concise answer supported by the text
  - qa_prompt: a closed-book prompt (question only, no passage)

No HTTP API/server is required. The output keeps one QA record per source train
text and mirrors the original triplet directory layout so it can be used as a
parallel QA dataset root.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from _hdbscan_pipeline_utils import data_dir, ensure_dir, load_json, now_utc_iso, write_json


DEFAULT_TRIPLET_DIR_CANDIDATES = (
    "wikitext_hdbscan_triplets",
    "wikitext_dbscan_triplets",
)
DEFAULT_INPUT_FILENAME = "train.json"
DEFAULT_OUTPUT_FILENAME = "train.json"
DEFAULT_MODEL = "openai/gpt-oss-20b"
MANIFEST_FILENAME = "run_manifest.json"
PROMPT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate QA prompts for each train.json record with a local vLLM model."
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
        "--qa_output_dir",
        type=str,
        default=None,
        help=(
            "QA output dir. Default: derive from input dir, e.g. "
            "<data_dir>/wikitext_hdbscan_triplets_qa."
        ),
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default=DEFAULT_INPUT_FILENAME,
        help="Per-triplet source filename to read (default: train.json).",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help="Per-triplet QA filename to write under the QA output dir (default: train.json).",
    )
    parser.add_argument(
        "--triplets",
        type=str,
        default=None,
        help=(
            'Specific triplets to process, e.g. "triplet_001 triplet_021" or '
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from already written QA outputs under --qa_output_dir.",
    )
    parser.add_argument(
        "--limit_per_triplet",
        type=int,
        default=None,
        help="Optional debug limit on number of train records processed per triplet.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Persist partial outputs after at least N new records (default: 5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="How many passages to generate per vLLM batch (default: 16).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Local model name/path for vLLM (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="vLLM dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size (default: 1).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction (default: 0.9).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Optional vLLM max model length override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for vLLM generation (default: 42).",
    )
    parser.add_argument(
        "--qa_language",
        type=str,
        default="English",
        help="Language for generated question and answer (default: English).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for QA generation (default: 0.2).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling value (default: 1.0).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Max generated tokens per QA pair (default: 1024).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Retries for malformed model outputs (default: 3).",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="",
        help=(
            "Optional chat template kwarg for GPT-OSS style models. "
            "Leave empty to reduce Harmony reasoning output (default: disabled)."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        dest="trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code when loading the model/tokenizer.",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading the model/tokenizer.",
    )
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM.",
    )
    parser.add_argument(
        "--use_structured_outputs",
        dest="use_structured_outputs",
        action="store_true",
        help="Use vLLM structured JSON outputs when available.",
    )
    parser.add_argument(
        "--disable_structured_outputs",
        dest="use_structured_outputs",
        action="store_false",
        help="Disable vLLM structured outputs and parse raw text only.",
    )
    parser.set_defaults(use_structured_outputs=True)
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


def resolve_qa_output_dir(explicit_path: Optional[str], triplet_input_dir: str) -> str:
    if explicit_path:
        return os.path.abspath(explicit_path)

    triplet_input_dir = os.path.abspath(triplet_input_dir)
    parent_dir = os.path.dirname(triplet_input_dir)
    base_name = os.path.basename(triplet_input_dir)
    output_base_name = base_name if base_name.endswith("_qa") else f"{base_name}_qa"
    return os.path.join(parent_dir, output_base_name)


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
    input_filename: str,
    start: int,
    end: Optional[int],
    selected_triplets: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    triplets: Dict[str, str] = {}
    for entry in os.listdir(triplet_input_dir):
        triplet_name = canonicalize_triplet_name(entry)
        triplet_dir = os.path.join(triplet_input_dir, entry)
        input_path = os.path.join(triplet_dir, input_filename)
        if not os.path.isdir(triplet_dir) or not os.path.isfile(input_path):
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


def read_text_records(path: str) -> List[Dict[str, str]]:
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


def read_existing_qa_records(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list in {path}")

    records: List[Dict[str, object]] = []
    required_fields = ("text", "question", "answer", "qa_prompt")
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(f"Expected item {idx} in {path} to be a JSON object")
        missing = [field for field in required_fields if not isinstance(item.get(field), str)]
        if missing:
            raise RuntimeError(
                f"Expected item {idx} in {path} to contain string fields: {', '.join(missing)}"
            )
        records.append(dict(item))
    return records


def write_records_json(path: str, data: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def validate_resume_records(
    existing_records: Sequence[Dict[str, object]], source_records: Sequence[Dict[str, str]], path: str
) -> None:
    if len(existing_records) > len(source_records):
        raise RuntimeError(
            f"Existing output has {len(existing_records)} records but source only has "
            f"{len(source_records)}: {path}"
        )
    for idx, record in enumerate(existing_records):
        if record.get("text") != source_records[idx]["text"]:
            raise RuntimeError(
                f"Existing output record {idx} in {path} does not match source text. "
                "Delete the stale output or rerun without --resume."
            )


def build_system_prompt(qa_language: str) -> str:
    return (
        "You create exactly one grounded question-answer pair from a passage.\n"
        "Return JSON only with keys \"question\" and \"answer\".\n"
        "Rules:\n"
        "- Use only information explicitly stated in the passage.\n"
        "- Write exactly one question and one answer.\n"
        "- The question must be specific, natural, and unambiguous.\n"
        "- Avoid yes/no questions.\n"
        "- Prefer key entities, events, dates, outcomes, titles, counts, or relationships.\n"
        "- The answer must be concise and fully supported by the passage.\n"
        "- If the passage is short, list-like, or fragmentary, still create the best factual QA pair.\n"
        f"- Write both question and answer in {qa_language}.\n"
        "- Do not include explanations, markdown, or extra keys."
    )


def build_user_prompt(text: str) -> str:
    return (
        "Passage:\n"
        f"{text}\n\n"
        "Return a JSON object in this exact shape:\n"
        '{"question": "...", "answer": "..."}'
    )


def build_qa_prompt(question: str) -> str:
    """Closed-book prompt: question only, no passage.

    The model must rely on parametric knowledge so that base-correct /
    unlearn-wrong labels genuinely reflect memorisation vs. forgetting.
    """
    return (
        f"Question:\n"
        f"{question}\n\n"
        "Answer:"
    )


def build_messages(text: str, qa_language: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(qa_language)},
        {"role": "user", "content": build_user_prompt(text)},
    ]


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def extract_first_json_object(text: str) -> Dict[str, object]:
    cleaned = strip_markdown_code_fence(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(cleaned)):
            ch = cleaned[end]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : end + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = cleaned.find("{", start + 1)

    raise RuntimeError(f"Failed to parse JSON object from model output: {text!r}")


def normalize_qa_pair(payload: Dict[str, object], qa_language: str) -> Tuple[str, str]:
    question = payload.get("question")
    answer = payload.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise RuntimeError(f"Model output missing non-empty string 'question': {payload}")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"Model output missing non-empty string 'answer': {payload}")

    normalized_question = re.sub(r"\s+", " ", question).strip()
    normalized_answer = re.sub(r"\s+", " ", answer).strip()

    if qa_language.strip().lower() == "english" and not normalized_question.endswith("?"):
        normalized_question = normalized_question.rstrip(".!") + "?"

    return normalized_question, normalized_answer


def build_structured_output_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["question", "answer"],
        "additionalProperties": False,
    }


def build_vllm_engine(args: argparse.Namespace):
    try:
        LLM = importlib.import_module("vllm").LLM
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'vllm'. Install it in the current Python environment first."
        ) from exc

    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": bool(args.trust_remote_code),
        "dtype": args.dtype,
        "seed": int(args.seed),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enforce_eager": bool(args.enforce_eager),
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = int(args.max_model_len)
    return LLM(**llm_kwargs)


def build_sampling_params(args: argparse.Namespace):
    try:
        vllm_module = importlib.import_module("vllm")
        sampling_params_module = importlib.import_module("vllm.sampling_params")
        SamplingParams = vllm_module.SamplingParams
        StructuredOutputsParams = sampling_params_module.StructuredOutputsParams
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'vllm'. Install it in the current Python environment first."
        ) from exc

    structured_outputs = None
    if args.use_structured_outputs:
        structured_outputs = StructuredOutputsParams(
            json=build_structured_output_schema(),
            disable_additional_properties=True,
        )

    return SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        seed=int(args.seed),
        structured_outputs=structured_outputs,
    )


def chat_template_kwargs(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    reasoning_effort = str(args.reasoning_effort).strip()
    if not reasoning_effort:
        return None
    return {"reasoning_effort": reasoning_effort}


def extract_request_output_text(request_output) -> str:
    outputs = getattr(request_output, "outputs", None)
    if not outputs:
        return ""
    text = getattr(outputs[0], "text", "")
    return text if isinstance(text, str) else ""


def extract_request_output_token_ids(request_output) -> List[int]:
    outputs = getattr(request_output, "outputs", None)
    if not outputs:
        return []
    token_ids = getattr(outputs[0], "token_ids", None)
    if token_ids is None:
        return []
    return [int(token_id) for token_id in token_ids]


def extract_harmony_final_content(token_ids: Sequence[int]) -> Optional[str]:
    if not token_ids:
        return None
    try:
        harmony_utils = importlib.import_module("vllm.entrypoints.openai.parser.harmony_utils")
        _reasoning, final_content, _is_tool_call = harmony_utils.parse_chat_output(list(token_ids))
    except Exception:
        return None
    if isinstance(final_content, str) and final_content.strip():
        return final_content.strip()
    return None


def extract_assistantfinal_segment(text: str) -> Optional[str]:
    cleaned = text.strip()
    if not cleaned:
        return None

    lower_cleaned = cleaned.lower()
    for marker in ("assistantfinal", "final answer", "final:"):
        idx = lower_cleaned.rfind(marker)
        if idx != -1:
            segment = cleaned[idx + len(marker) :].lstrip(" \t\r\n:=")
            return segment or None
    return None


def decode_jsonish_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except Exception:
        decoded = value
        replacements = (
            ('\\"', '"'),
            ("\\n", "\n"),
            ("\\t", "\t"),
            ("\\r", "\r"),
            ("\\\\", "\\"),
        )
        for old, new in replacements:
            decoded = decoded.replace(old, new)
        return decoded


def extract_partial_qa_payload(text: str) -> Optional[Dict[str, object]]:
    candidate = extract_assistantfinal_segment(text) or text
    if not candidate:
        return None

    question_match = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate, flags=re.DOTALL)
    if not question_match:
        return None

    answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate, flags=re.DOTALL)
    answer_is_partial = False
    if not answer_match:
        answer_match = re.search(r'"answer"\s*:\s*"([\s\S]*)$', candidate, flags=re.DOTALL)
        answer_is_partial = answer_match is not None
    if not answer_match:
        return None

    question = decode_jsonish_string(question_match.group(1)).strip()
    answer = decode_jsonish_string(answer_match.group(1)).strip()
    if answer_is_partial:
        answer = answer.rstrip('"} \t\r\n')

    if not question or not answer:
        return None
    return {"question": question, "answer": answer}


def parse_qa_pair_from_output(
    raw_text: str,
    token_ids: Sequence[int],
    qa_language: str,
) -> Tuple[str, str]:
    candidates: List[str] = []
    seen = set()
    for candidate in (
        extract_harmony_final_content(token_ids),
        extract_assistantfinal_segment(raw_text),
        raw_text,
    ):
        if not candidate:
            continue
        normalized_candidate = candidate.strip()
        if normalized_candidate and normalized_candidate not in seen:
            candidates.append(normalized_candidate)
            seen.add(normalized_candidate)

    errors: List[str] = []
    for candidate in candidates:
        try:
            qa_payload = extract_first_json_object(candidate)
            return normalize_qa_pair(qa_payload, qa_language)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    for candidate in candidates:
        partial_payload = extract_partial_qa_payload(candidate)
        if partial_payload is not None:
            return normalize_qa_pair(partial_payload, qa_language)

    raise RuntimeError(
        "Failed to parse QA payload from model output. "
        f"Errors: {errors or ['no parse candidates']}. Raw output: {raw_text!r}"
    )


def call_vllm_chat(
    llm,
    sampling_params,
    args: argparse.Namespace,
    message_batches: Sequence[Sequence[Dict[str, str]]],
) -> List[object]:
    outputs = llm.chat(
        list(message_batches),
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template_kwargs=chat_template_kwargs(args),
    )
    if len(outputs) != len(message_batches):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(message_batches)} prompts."
        )
    return list(outputs)


def generate_single_qa_pair(
    llm,
    sampling_params,
    args: argparse.Namespace,
    text: str,
) -> Tuple[str, str]:
    messages = [build_messages(text, args.qa_language)]
    last_error: Optional[Exception] = None
    for attempt in range(1, int(args.max_retries) + 1):
        request_output = call_vllm_chat(llm, sampling_params, args, messages)[0]
        raw_output = extract_request_output_text(request_output)
        token_ids = extract_request_output_token_ids(request_output)
        try:
            return parse_qa_pair_from_output(raw_output, token_ids, args.qa_language)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < int(args.max_retries):
                print(
                    f"    malformed output (attempt {attempt}/{args.max_retries}) -> retry",
                    flush=True,
                )
    raise RuntimeError(f"Failed to generate valid QA JSON after {args.max_retries} attempts: {last_error}")


def generate_qa_pairs_batch(
    llm,
    sampling_params,
    args: argparse.Namespace,
    texts: Sequence[str],
) -> List[Tuple[str, str]]:
    message_batches = [build_messages(text, args.qa_language) for text in texts]
    request_outputs = call_vllm_chat(llm, sampling_params, args, message_batches)

    pairs: List[Tuple[str, str]] = []
    for idx, (text, request_output) in enumerate(zip(texts, request_outputs)):
        raw_output = extract_request_output_text(request_output)
        token_ids = extract_request_output_token_ids(request_output)
        try:
            pairs.append(parse_qa_pair_from_output(raw_output, token_ids, args.qa_language))
        except Exception as exc:  # noqa: BLE001
            print(
                f"    batch item {idx + 1}/{len(texts)} produced invalid JSON ({exc}); "
                "retrying individually",
                flush=True,
            )
            pairs.append(generate_single_qa_pair(llm, sampling_params, args, text))
    return pairs


def build_output_record(
    triplet_name: str,
    source_meta: Dict[str, object],
    source_train_index: int,
    text: str,
    question: str,
    answer: str,
) -> Dict[str, object]:
    return {
        "triplet": triplet_name,
        "cluster_label": source_meta.get("cluster_label"),
        "domain": source_meta.get("domain"),
        "domain_triplet_index": source_meta.get("domain_triplet_index"),
        "source_train_index": int(source_train_index),
        "text": text,
        "question": question,
        "answer": answer,
        "qa_prompt": build_qa_prompt(question),
    }


def build_manifest(
    existing_manifest: Dict[str, object],
    triplet_input_dir: str,
    triplet_manifest_path: str,
    qa_output_dir: str,
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
        "input_filename": args.input_filename,
    }
    manifest["qa_generation"] = {
        "backend": "vllm",
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len) if args.max_model_len is not None else None,
        "trust_remote_code": bool(args.trust_remote_code),
        "enforce_eager": bool(args.enforce_eager),
        "batch_size": int(args.batch_size),
        "qa_language": args.qa_language,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "seed": int(args.seed),
        "max_retries": int(args.max_retries),
        "reasoning_effort": str(args.reasoning_effort).strip() or None,
        "prompt_version": PROMPT_VERSION,
        "use_structured_outputs": bool(args.use_structured_outputs),
    }
    manifest["selection"] = {
        "triplets": list(args.triplet_list) if args.triplet_list else None,
        "start": int(args.start),
        "end": int(args.end) if args.end is not None else None,
        "resume": bool(args.resume),
        "limit_per_triplet": int(args.limit_per_triplet) if args.limit_per_triplet else None,
    }
    manifest["outputs"] = {
        "qa_output_dir": qa_output_dir,
        "per_triplet_output_filename": args.output_filename,
        "manifest_json": output_manifest_path,
    }
    manifest["stats"] = {
        "n_triplets": len(written_triplets),
        "total_source_train_records": sum(int(item["source_train_size"]) for item in written_triplets),
        "total_qa_records_written": sum(int(item["qa_records_written"]) for item in written_triplets),
        "total_new_records_generated": sum(int(item["new_records_generated"]) for item in written_triplets),
    }
    manifest["triplets"] = written_triplets
    manifest["pipeline"] = {
        "step": "qa",
        "status": "completed",
    }
    return manifest


def free_vllm_engine(llm) -> None:
    del llm
    gc.collect()
    try:
        torch = importlib.import_module("torch")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    args.triplet_list = parse_triplet_selection(args.triplets)

    if not args.model:
        raise RuntimeError("Missing --model.")
    if args.start <= 0:
        raise RuntimeError("Require --start > 0")
    if args.end is not None and args.end < args.start:
        raise RuntimeError("Require --end >= --start")
    if args.limit_per_triplet is not None and args.limit_per_triplet <= 0:
        raise RuntimeError("Require --limit_per_triplet > 0 when provided")
    if args.save_every <= 0:
        raise RuntimeError("Require --save_every > 0")
    if args.batch_size <= 0:
        raise RuntimeError("Require --batch_size > 0")
    if args.max_tokens <= 0:
        raise RuntimeError("Require --max_tokens > 0")
    if args.temperature < 0:
        raise RuntimeError("Require --temperature >= 0")
    if args.top_p <= 0 or args.top_p > 1:
        raise RuntimeError("Require 0 < --top_p <= 1")
    if args.max_retries <= 0:
        raise RuntimeError("Require --max_retries > 0")
    if args.tensor_parallel_size <= 0:
        raise RuntimeError("Require --tensor_parallel_size > 0")
    if args.gpu_memory_utilization <= 0 or args.gpu_memory_utilization > 1:
        raise RuntimeError("Require 0 < --gpu_memory_utilization <= 1")
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise RuntimeError("Require --max_model_len > 0 when provided")

    triplet_input_dir = resolve_triplet_input_dir(args.triplet_input_dir)
    qa_output_dir = ensure_dir(resolve_qa_output_dir(args.qa_output_dir, triplet_input_dir))
    triplet_manifest_path = os.path.join(triplet_input_dir, MANIFEST_FILENAME)
    output_manifest_path = os.path.join(qa_output_dir, MANIFEST_FILENAME)

    existing_output_manifest = load_existing_manifest(output_manifest_path)
    source_triplet_manifest = load_existing_manifest(triplet_manifest_path)
    triplet_meta_index = build_triplet_meta_index(source_triplet_manifest)

    triplets = discover_triplets(
        triplet_input_dir=triplet_input_dir,
        input_filename=args.input_filename,
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

    print("Generating QA train splits from triplet train.json files ...")
    print(f"  triplet_input_dir:      {triplet_input_dir}")
    print(f"  qa_output_dir:          {qa_output_dir}")
    print(f"  backend:                vllm")
    print(f"  model:                  {args.model}")
    print(f"  dtype:                  {args.dtype}")
    print(f"  tensor_parallel_size:   {args.tensor_parallel_size}")
    print(f"  gpu_memory_utilization: {args.gpu_memory_utilization}")
    print(f"  batch_size:             {args.batch_size}")
    print(f"  input_filename:         {args.input_filename}")
    print(f"  output_filename:        {args.output_filename}")
    print(f"  triplets:               {len(triplets)} ({selection_label})")
    if args.limit_per_triplet:
        print(f"  limit_per_triplet:      {args.limit_per_triplet}")
    if args.resume:
        print("  resume:                 enabled")
    print("Loading local vLLM engine ...", flush=True)
    sampling_params = build_sampling_params(args)
    llm = build_vllm_engine(args)
    print("  vLLM ready.", flush=True)

    written_triplets: List[Dict[str, object]] = []
    total_triplets = len(triplets)
    try:
        for triplet_offset, (triplet_name, triplet_dir) in enumerate(triplets, start=1):
            source_path = os.path.join(triplet_dir, args.input_filename)
            source_records = read_text_records(source_path)
            if args.limit_per_triplet is not None:
                source_records = source_records[: args.limit_per_triplet]

            output_triplet_dir = ensure_dir(os.path.join(qa_output_dir, triplet_name))
            output_path = os.path.join(output_triplet_dir, args.output_filename)
            source_meta = triplet_meta_index.get(triplet_name, {})

            existing_records: List[Dict[str, object]] = []
            if args.resume and os.path.isfile(output_path):
                existing_records = read_existing_qa_records(output_path)
                validate_resume_records(existing_records, source_records, output_path)

            total_records = len(source_records)
            if len(existing_records) == total_records and total_records > 0:
                print(
                    f"  [{triplet_offset:3d}/{total_triplets}] {triplet_name}: "
                    f"already complete ({total_records} records)"
                )
                new_records_generated = 0
                qa_records = existing_records
            else:
                print(
                    f"  [{triplet_offset:3d}/{total_triplets}] {triplet_name}: "
                    f"source={total_records} existing={len(existing_records)}"
                )

                qa_records = list(existing_records)
                new_records_generated = 0
                new_records_since_last_save = 0
                source_idx = len(existing_records)
                while source_idx < total_records:
                    batch_end = min(source_idx + int(args.batch_size), total_records)
                    batch_records = source_records[source_idx:batch_end]
                    batch_texts = [record["text"] for record in batch_records]
                    batch_pairs = generate_qa_pairs_batch(
                        llm=llm,
                        sampling_params=sampling_params,
                        args=args,
                        texts=batch_texts,
                    )
                    for local_idx, (text, (question, answer)) in enumerate(
                        zip(batch_texts, batch_pairs)
                    ):
                        qa_records.append(
                            build_output_record(
                                triplet_name=triplet_name,
                                source_meta=source_meta,
                                source_train_index=source_idx + local_idx,
                                text=text,
                                question=question,
                                answer=answer,
                            )
                        )

                    batch_size_written = len(batch_pairs)
                    new_records_generated += batch_size_written
                    new_records_since_last_save += batch_size_written
                    source_idx = batch_end

                    if (
                        new_records_since_last_save >= int(args.save_every)
                        or len(qa_records) == total_records
                    ):
                        write_records_json(output_path, qa_records)
                        print(
                            f"    saved {len(qa_records)}/{total_records} QA records "
                            f"to {output_path}",
                            flush=True,
                        )
                        new_records_since_last_save = 0

                if not os.path.isfile(output_path):
                    write_records_json(output_path, qa_records)

            written_triplets.append(
                {
                    "name": triplet_name,
                    "cluster_label": source_meta.get("cluster_label"),
                    "domain": source_meta.get("domain"),
                    "domain_triplet_index": source_meta.get("domain_triplet_index"),
                    "source_input_json": source_path,
                    "output_train_json": output_path,
                    "source_train_size": total_records,
                    "qa_records_written": len(qa_records),
                    "new_records_generated": new_records_generated,
                }
            )
    finally:
        free_vllm_engine(llm)

    manifest = build_manifest(
        existing_manifest=existing_output_manifest,
        triplet_input_dir=triplet_input_dir,
        triplet_manifest_path=triplet_manifest_path,
        qa_output_dir=qa_output_dir,
        output_manifest_path=output_manifest_path,
        args=args,
        written_triplets=written_triplets,
    )
    write_json(output_manifest_path, manifest)

    print("\nDone.")
    print(f"  qa_output_dir:        {qa_output_dir}")
    print(f"  triplets_written:     {manifest['stats']['n_triplets']}")
    print(f"  total_source_records: {manifest['stats']['total_source_train_records']}")
    print(f"  total_qa_records:     {manifest['stats']['total_qa_records_written']}")
    print(f"  new_records_created:  {manifest['stats']['total_new_records_generated']}")
    print(f"  manifest:             {output_manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
