#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_ga_wikitext_batch.sh [START] [END] [PARALLEL]
#   e.g. ./run_ga_wikitext_batch.sh              # all 100, 1 at a time
#        ./run_ga_wikitext_batch.sh 11 100        # triplet_011–100, 1 at a time
#        ./run_ga_wikitext_batch.sh 1 100 2       # all 100, 2 at a time
#
# Runs GradAscent unlearning on WikiText triplets using Llama-3.2-1B-Instruct.
# Each triplet's train.json = forget set (variable size), val.json = retain set.
# Checkpoints are saved under THIS directory's saves/ (not inside open-unlearning).

if [[ "${CONDA_DEFAULT_ENV:-}" != "unlearning" ]]; then
  echo "Please run: conda activate unlearning"
  exit 1
fi

START="${1:-1}"
END="${2:-100}"
PARALLEL="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPEN_UNLEARN_DIR="$(cd "${SCRIPT_DIR}/../open-unlearning" && pwd)"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../data/wikitext" && pwd)"

SAVE_ROOT="${SCRIPT_DIR}/saves/unlearn"
LOG_DIR="${SCRIPT_DIR}/logs_wikitext"
mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"

MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-meta-llama/${MODEL}}"
TRAINER="GradAscent"

BS="${BS:-4}"
GAS="${GAS:-4}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
GPU="${GPU:-0}"

echo "============================================"
echo "  WikiText GradAscent Batch: ${MODEL}"
echo "  Range:    triplet_$(printf '%03d' ${START}) – triplet_$(printf '%03d' ${END})"
echo "  Parallel: ${PARALLEL} jobs"
echo "  BS=${BS}  GAS=${GAS}  GPU=${GPU}"
echo "  Data:     ${DATA_ROOT}"
echo "  Saves:    ${SAVE_ROOT}"
echo "  Logs:     ${LOG_DIR}/"
echo "============================================"

TOTAL=$(( END - START + 1 ))
DONE=0
FAIL=0
SKIP=0

run_one() {
  local idx=$1
  local triplet_tag=$(printf "triplet_%03d" "${idx}")
  local task_name="wikitext_${MODEL}_${triplet_tag}_${TRAINER}"
  local forget_json="${DATA_ROOT}/${triplet_tag}/train.json"
  local retain_json="${DATA_ROOT}/${triplet_tag}/val.json"
  local log_file="${LOG_DIR}/${triplet_tag}.log"

  if [[ ! -f "${forget_json}" ]]; then
    echo "  [MISS] ${triplet_tag} – train.json not found"
    return 1
  fi

  if [[ -d "${SAVE_ROOT}/${task_name}" && -f "${SAVE_ROOT}/${task_name}/model.safetensors" ]]; then
    echo "  [SKIP] ${triplet_tag} – already exists"
    return 2
  fi

  cd "${OPEN_UNLEARN_DIR}"

  CUDA_VISIBLE_DEVICES=${GPU} python src/train.py \
    --config-name=unlearn.yaml \
    "experiment=unlearn/wikitext/default.yaml" \
    "model=${MODEL}" \
    "trainer=${TRAINER}" \
    "task_name=${task_name}" \
    "paths.output_dir=${SAVE_ROOT}/${task_name}" \
    "model.model_args.pretrained_model_name_or_path=${BASE_MODEL}" \
    "model.model_args.attn_implementation=${ATTN_IMPL}" \
    "trainer.args.per_device_train_batch_size=${BS}" \
    "trainer.args.gradient_accumulation_steps=${GAS}" \
    "data.forget.WikiText_forget.args.hf_args.path=json" \
    "+data.forget.WikiText_forget.args.hf_args.data_files=${forget_json}" \
    "data.retain.WikiText_retain.args.hf_args.path=json" \
    "+data.retain.WikiText_retain.args.hf_args.data_files=${retain_json}" \
    > "${log_file}" 2>&1
}

idx=${START}
while (( idx <= END )); do
  pids=()
  batch_idxs=()
  for (( i=0; i<PARALLEL && idx<=END; i++, idx++ )); do
    triplet_tag=$(printf "triplet_%03d" "${idx}")
    task_name="wikitext_${MODEL}_${triplet_tag}_${TRAINER}"

    if [[ -d "${SAVE_ROOT}/${task_name}" && -f "${SAVE_ROOT}/${task_name}/model.safetensors" ]]; then
      echo "  [SKIP] ${triplet_tag} – already trained"
      (( SKIP++ )) || true
      (( i-- )) || true
      continue
    fi

    echo "  [START] ${triplet_tag}"
    run_one "${idx}" &
    pids+=($!)
    batch_idxs+=("${idx}")
  done

  if (( ${#pids[@]} == 0 )); then
    continue
  fi

  for j in "${!pids[@]}"; do
    pid=${pids[$j]}
    bi=${batch_idxs[$j]}
    tt=$(printf "triplet_%03d" "${bi}")
    if wait "${pid}"; then
      echo "  [DONE]  ${tt}"
      (( DONE++ )) || true
    else
      code=$?
      if (( code == 2 )); then
        echo "  [SKIP]  ${tt}"
        (( SKIP++ )) || true
      else
        echo "  [FAIL]  ${tt}  (exit ${code}, see ${LOG_DIR}/${tt}.log)"
        (( FAIL++ )) || true
      fi
    fi
  done

  echo "  --- batch done (progress: $((DONE+SKIP+FAIL))/${TOTAL}) ---"
done

echo
echo "============================================"
echo "  Finished: ${DONE} trained, ${SKIP} skipped, ${FAIL} failed"
echo "  Saves:    ${SAVE_ROOT}"
echo "  Logs:     ${LOG_DIR}/"
echo "============================================"

if (( FAIL > 0 )); then
  echo
  echo "  Failed triplets:"
  for f in "${LOG_DIR}"/*.log; do
    tag=$(basename "${f}" .log)
    if ! grep -q "train_runtime" "${f}" 2>/dev/null; then
      echo "    ${tag}  →  ${f}"
    fi
  done
  exit 1
fi
