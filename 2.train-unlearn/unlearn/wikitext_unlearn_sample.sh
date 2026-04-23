#!/bin/bash
#
# Quick single-triplet unlearning run (triplet_001) to verify GradAscent convergence.
# Logs go to both terminal and log file so you can watch loss in real time.
#
# Usage:
#   ./wikitext_unlearn_sample.sh
#   ./wikitext_unlearn_sample.sh --gpu 0 --epochs 3 --lr 1e-5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_UNLEARN_DIR="/media/volume/llm/unlearning/2.train-unlearn/open-unlearning"
TRIPLET_ROOT="/media/volume/llm/unlearning/1.data-preparation/data/wikitext_hdbscan_triplets"

SAVE_DIR="${SCRIPT_DIR}/saves/wikitext_unlearn_sample"
LOG_DIR="${SCRIPT_DIR}/logs_wikitext_unlearn_sample"

GPU="0"
MODEL="Llama-3.1-8B-Instruct"
TRAINER="GradAscent"
TRIPLET="triplet_001"
BS=1
GAS=1
EPOCHS=10
LR="5e-4"
ATTN_IMPL="sdpa"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)       GPU="$2";       shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --trainer)   TRAINER="$2";   shift 2 ;;
        --triplet)   TRIPLET="$2";   shift 2 ;;
        --bs)        BS="$2";        shift 2 ;;
        --gas)       GAS="$2";       shift 2 ;;
        --epochs)    EPOCHS="$2";    shift 2 ;;
        --lr)        LR="$2";        shift 2 ;;
        --attn_impl) ATTN_IMPL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

FORGET_FILE="${TRIPLET_ROOT}/${TRIPLET}/train.json"
RETAIN_FILE="${TRIPLET_ROOT}/${TRIPLET}/validation.json"

for f in "${FORGET_FILE}" "${RETAIN_FILE}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: missing ${f}"
        exit 1
    fi
done

TASK_NAME="wikitext_${MODEL}_${TRIPLET}_${TRAINER}"
OUTPUT_DIR="${SAVE_DIR}/${TASK_NAME}"
TRAIN_LOG="${LOG_DIR}/${TASK_NAME}.log"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

echo "============================================"
echo "Sample Unlearning Run"
echo "  GPU:      ${GPU}"
echo "  Model:    ${MODEL}"
echo "  Trainer:  ${TRAINER}"
echo "  Triplet:  ${TRIPLET}"
echo "  BS:       ${BS}  GAS: ${GAS}"
echo "  Epochs:   ${EPOCHS}"
echo "  LR:       ${LR}"
echo "  Attn:     ${ATTN_IMPL}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Log:      ${TRAIN_LOG}"
echo "============================================"

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
    trainer.args.logging_steps=1 \
    2>&1 | tee "${TRAIN_LOG}"

echo ""
echo "============================================"
echo "Done. Log saved to ${TRAIN_LOG}"
echo "============================================"
