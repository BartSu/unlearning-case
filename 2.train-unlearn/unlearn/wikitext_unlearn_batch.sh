#!/bin/bash
#
# Batch unlearning over all triplets (skips those whose checkpoint already
# exists). Failure on one triplet does NOT abort the batch.
#
# Usage:
#   ./wikitext_unlearn_batch.sh                        # all 101 triplets
#   ./wikitext_unlearn_batch.sh --only 002,012,022     # explicit subset
#   ./wikitext_unlearn_batch.sh --dry-run              # print plan, no training
#   ./wikitext_unlearn_batch.sh --gpu 0 --epochs 10 --lr 5e-4
#
# Outputs
#   saves/wikitext_unlearn/<TASK>/        per-triplet checkpoint
#   logs_wikitext_unlearn_batch/<TASK>.log
#   logs_wikitext_unlearn_batch/_summary.csv  (triplet, status, wall_seconds)

set -uo pipefail  # no -e: we want to continue on per-triplet failure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_UNLEARN_DIR="/media/volume/llm/unlearning/2.train-unlearn/open-unlearning"
TRIPLET_ROOT="/media/volume/llm/unlearning/1.data-preparation/data/wikitext_hdbscan_triplets"

SAVE_DIR="${SCRIPT_DIR}/saves/wikitext_unlearn"
LOG_DIR="${SCRIPT_DIR}/logs_wikitext_unlearn_batch"
SUMMARY_CSV="${LOG_DIR}/_summary.csv"

GPU="0"
MODEL="Llama-3.1-8B-Instruct"
TRAINER="GradAscent"
BS=1
GAS=1
EPOCHS=10
LR="5e-4"
ATTN_IMPL="sdpa"
ONLY=""        # comma-separated numeric suffixes, e.g. "002,012"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)       GPU="$2";       shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --trainer)   TRAINER="$2";   shift 2 ;;
        --bs)        BS="$2";        shift 2 ;;
        --gas)       GAS="$2";       shift 2 ;;
        --epochs)    EPOCHS="$2";    shift 2 ;;
        --lr)        LR="$2";        shift 2 ;;
        --attn_impl) ATTN_IMPL="$2"; shift 2 ;;
        --only)      ONLY="$2";      shift 2 ;;
        --dry-run)   DRY_RUN=1;      shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
    echo "triplet,status,wall_seconds,task_name" > "${SUMMARY_CSV}"
fi

# Build triplet list
mapfile -t ALL_TRIPLETS < <(ls -1 "${TRIPLET_ROOT}" | grep -E '^triplet_[0-9]+$' | sort)

if [[ -n "${ONLY}" ]]; then
    IFS=',' read -r -a WANT <<< "${ONLY}"
    FILTERED=()
    for tr in "${ALL_TRIPLETS[@]}"; do
        suffix="${tr#triplet_}"
        for w in "${WANT[@]}"; do
            if [[ "${suffix}" == "${w}" ]]; then
                FILTERED+=("${tr}"); break
            fi
        done
    done
    ALL_TRIPLETS=("${FILTERED[@]}")
fi

echo "============================================"
echo "Batch plan"
echo "  GPU:       ${GPU}"
echo "  Model:     ${MODEL}"
echo "  Trainer:   ${TRAINER}"
echo "  Candidates: ${#ALL_TRIPLETS[@]} triplets"
echo "  SAVE_DIR:  ${SAVE_DIR}"
echo "  LOG_DIR:   ${LOG_DIR}"
echo "============================================"

N_SKIP=0; N_RUN=0; N_FAIL=0
for TRIPLET in "${ALL_TRIPLETS[@]}"; do
    TASK_NAME="wikitext_${MODEL}_${TRIPLET}_${TRAINER}"
    OUTPUT_DIR="${SAVE_DIR}/${TASK_NAME}"
    FORGET_FILE="${TRIPLET_ROOT}/${TRIPLET}/train.json"
    RETAIN_FILE="${TRIPLET_ROOT}/${TRIPLET}/validation.json"

    if [[ ! -f "${FORGET_FILE}" || ! -f "${RETAIN_FILE}" ]]; then
        echo "[SKIP-missing] ${TRIPLET}"
        echo "${TRIPLET},missing,0,${TASK_NAME}" >> "${SUMMARY_CSV}"
        N_SKIP=$((N_SKIP+1)); continue
    fi

    # Skip if checkpoint already exists and non-empty
    if [[ -d "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
        echo "[SKIP-done]    ${TRIPLET}"
        echo "${TRIPLET},skipped,0,${TASK_NAME}" >> "${SUMMARY_CSV}"
        N_SKIP=$((N_SKIP+1)); continue
    fi

    echo "[RUN]          ${TRIPLET}  (epochs=${EPOCHS} lr=${LR})"
    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi

    TRAIN_LOG="${LOG_DIR}/${TASK_NAME}.log"
    T0=$(date +%s)

    (
        cd "${OPEN_UNLEARN_DIR}"
        CUDA_VISIBLE_DEVICES="${GPU}" python src/train.py \
            --config-name=unlearn.yaml \
            experiment=unlearn/wikitext/default \
            model="${MODEL}" \
            trainer="${TRAINER}" \
            triplet_name="${TRIPLET}" \
            task_name="${TASK_NAME}" \
            paths.output_dir="${OUTPUT_DIR}" \
            model.model_args.attn_implementation="${ATTN_IMPL}" \
            data.forget.WIKITEXT.args.hf_args.data_files="${FORGET_FILE}" \
            data.retain.WIKITEXT.args.hf_args.data_files="${RETAIN_FILE}" \
            trainer.args.per_device_train_batch_size="${BS}" \
            trainer.args.gradient_accumulation_steps="${GAS}" \
            trainer.args.num_train_epochs="${EPOCHS}" \
            trainer.args.learning_rate="${LR}" \
            trainer.args.logging_steps=1
    ) > "${TRAIN_LOG}" 2>&1
    RC=$?
    DUR=$(( $(date +%s) - T0 ))

    if [[ ${RC} -eq 0 ]]; then
        echo "  → ok (${DUR}s)"
        echo "${TRIPLET},ok,${DUR},${TASK_NAME}" >> "${SUMMARY_CSV}"
        N_RUN=$((N_RUN+1))
    else
        echo "  → FAIL rc=${RC} (see ${TRAIN_LOG})"
        echo "${TRIPLET},fail_rc${RC},${DUR},${TASK_NAME}" >> "${SUMMARY_CSV}"
        N_FAIL=$((N_FAIL+1))
        # continue to next triplet
    fi
done

echo "============================================"
echo "Batch done:  run=${N_RUN}  skipped=${N_SKIP}  failed=${N_FAIL}"
echo "Summary: ${SUMMARY_CSV}"
echo "============================================"
