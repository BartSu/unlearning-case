#!/bin/bash
#
# WikiText triplet unlearning using the open-unlearning framework.
# By default this discovers triplets under data/wikitext_hdbscan_triplets and
# uses:
#   - train.json as forget data
#   - validation.json as retain data
#
# Notes:
# - For GradAscent, the retain split is not used in the loss, but it is still
#   provided because the unlearning data wrapper expects both forget/retain
#   datasets to exist.
#
# Usage:
#   ./wikitext_unlearn.sh
#   ./wikitext_unlearn.sh --triplets "triplet_001 triplet_002"
#   ./wikitext_unlearn.sh --start 1 --end 4
#   ./wikitext_unlearn.sh --gpu 0 --epochs 1 --lr 1e-5
#   ./wikitext_unlearn.sh --dry_run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_UNLEARN_DIR="/media/volume/llm/unlearning/data-preparation/open-unlearning"
TRIPLET_ROOT="/media/volume/llm/unlearning/data-preparation/data/wikitext_hdbscan_triplets"

SAVE_DIR="${SCRIPT_DIR}/saves/wikitext_unlearn"
LOG_DIR="${SCRIPT_DIR}/logs_wikitext_unlearn"

GPU="0"
MODELS=("Llama-3.1-8B-Instruct")
# MODELS=("Llama-3.1-8B") # base model ppl is lower, open-unlearning only support llama instruct models
TRAINERS=("GradAscent")
TRIPLETS=()
START=1
END=0
BS=1
GAS=8
EPOCHS=1
LR="1e-5"
ATTN_IMPL="sdpa"
DRY_RUN=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  --gpu GPU_IDS              CUDA_VISIBLE_DEVICES (default: 0)"
    echo "  --models \"m1 m2\"           Model config names"
    echo "  --trainers \"t1 t2\"         Trainer names"
    echo "  --triplets \"t1 t2\"         Specific triplets to run"
    echo "  --triplet_root PATH        Root dir containing triplet_* folders"
    echo "  --start N                  1-based start index into discovered triplets"
    echo "  --end N                    1-based end index into discovered triplets"
    echo "  --bs N                     Per-device batch size (default: 1)"
    echo "  --gas N                    Gradient accumulation steps (default: 8)"
    echo "  --epochs N                 Training epochs (default: 1)"
    echo "  --lr RATE                  Learning rate (default: 1e-5)"
    echo "  --attn_impl NAME           Attention backend (default: sdpa)"
    echo "  --dry_run                  Print selected runs without training"
    exit 1
}

discover_triplets() {
    local discovered=()
    local dir
    shopt -s nullglob
    for dir in "${TRIPLET_ROOT}"/triplet_*; do
        [[ -d "${dir}" && -f "${dir}/train.json" && -f "${dir}/validation.json" ]] || continue
        discovered+=("$(basename "${dir}")")
    done
    shopt -u nullglob
    printf '%s\n' "${discovered[@]}"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        --models) read -ra MODELS <<< "$2"; shift 2 ;;
        --trainers) read -ra TRAINERS <<< "$2"; shift 2 ;;
        --triplets) read -ra TRIPLETS <<< "$2"; shift 2 ;;
        --triplet_root) TRIPLET_ROOT="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --end) END="$2"; shift 2 ;;
        --bs) BS="$2"; shift 2 ;;
        --gas) GAS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --attn_impl) ATTN_IMPL="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

mapfile -t DISCOVERED_TRIPLETS < <(discover_triplets)
if [[ ${#DISCOVERED_TRIPLETS[@]} -eq 0 ]]; then
    echo "No triplets found under ${TRIPLET_ROOT}"
    exit 1
fi

if [[ ${#TRIPLETS[@]} -eq 0 ]]; then
    if [[ ${END} -eq 0 ]]; then
        END=${#DISCOVERED_TRIPLETS[@]}
    fi
    if (( START < 1 || END < START || END > ${#DISCOVERED_TRIPLETS[@]} )); then
        echo "Invalid --start/--end range for ${#DISCOVERED_TRIPLETS[@]} discovered triplets"
        exit 1
    fi
    TRIPLETS=("${DISCOVERED_TRIPLETS[@]:$((START-1)):$((END-START+1))}")
fi

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

echo "============================================"
echo "WikiText Triplet Unlearning"
echo "  GPU:          ${GPU}"
echo "  Models:       ${MODELS[*]}"
echo "  Trainers:     ${TRAINERS[*]}"
echo "  Triplets:     ${#TRIPLETS[@]} (${TRIPLETS[0]} .. ${TRIPLETS[-1]})"
echo "  Triplet root: ${TRIPLET_ROOT}"
echo "  Save:         ${SAVE_DIR}"
echo "  Logs:         ${LOG_DIR}"
echo "  Attn impl:    ${ATTN_IMPL}"
echo "  Dry run:      ${DRY_RUN}"
echo "============================================"

done_count=0
skip_count=0
fail_count=0

cd "${OPEN_UNLEARN_DIR}"

for model in "${MODELS[@]}"; do
    for trainer in "${TRAINERS[@]}"; do
        for triplet in "${TRIPLETS[@]}"; do
            task_name="wikitext_${model}_${triplet}_${trainer}"
            output_dir="${SAVE_DIR}/${task_name}"
            train_log="${LOG_DIR}/${task_name}.log"
            forget_file="${TRIPLET_ROOT}/${triplet}/train.json"
            retain_file="${TRIPLET_ROOT}/${triplet}/validation.json"

            if [[ ! -f "${forget_file}" ]]; then
                echo "[SKIP] ${task_name}: missing ${forget_file}"
                ((skip_count++)) || true
                continue
            fi

            if [[ ! -f "${retain_file}" ]]; then
                echo "[SKIP] ${task_name}: missing ${retain_file}"
                ((skip_count++)) || true
                continue
            fi

            if [[ -f "${output_dir}/model.safetensors" || -f "${output_dir}/model.safetensors.index.json" ]]; then
                echo "[SKIP-TRAIN] ${task_name} (model exists)"
                ((skip_count++)) || true
                continue
            fi

            if ${DRY_RUN}; then
                echo "[DRY-RUN] ${task_name}"
                echo "          forget=${forget_file}"
                echo "          retain=${retain_file}"
                ((done_count++)) || true
                continue
            fi

            echo "[TRAIN] ${task_name}"
            if CUDA_VISIBLE_DEVICES="${GPU}" python src/train.py \
                --config-name=unlearn.yaml \
                experiment=unlearn/wikitext/default \
                model="${model}" \
                trainer="${trainer}" \
                triplet_name="${triplet}" \
                task_name="${task_name}" \
                paths.output_dir="${output_dir}" \
                model.model_args.attn_implementation="${ATTN_IMPL}" \
                data.forget.WIKITEXT.args.hf_args.data_files="${forget_file}" \
                data.retain.WIKITEXT.args.hf_args.data_files="${retain_file}" \
                trainer.args.per_device_train_batch_size="${BS}" \
                trainer.args.gradient_accumulation_steps="${GAS}" \
                trainer.args.num_train_epochs="${EPOCHS}" \
                trainer.args.learning_rate="${LR}" \
                > "${train_log}" 2>&1; then
                echo "[DONE] ${task_name}"
                ((done_count++)) || true
            else
                echo "[FAIL] ${task_name} (see ${train_log})"
                ((fail_count++)) || true
                continue
            fi
        done
    done
done

echo ""
echo "============================================"
echo "Finished."
echo "  Done:  ${done_count}"
echo "  Skip:  ${skip_count}"
echo "  Fail:  ${fail_count}"
echo "  Saves: ${SAVE_DIR}"
echo "  Logs:  ${LOG_DIR}"
echo "============================================"
