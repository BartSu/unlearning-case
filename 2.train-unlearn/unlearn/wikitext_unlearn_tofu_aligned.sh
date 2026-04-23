#!/bin/bash
#
# TOFU-aligned GradAscent unlearning (single-GPU H100).
#
# Training config:
#   per_device_batch=16, grad_accum=4   -> effective_batch = 64
#   num_epochs=5, max_steps=3 (hard-pinned)
#   lr=1e-5, scheduler=linear, warmup_steps=1
#   optim=paged_adamw_32bit, weight_decay=0.01, bf16=True
#   samples_seen = 50 * 5 = 250 per triplet
#
# Usage:
#   ./wikitext_unlearn_tofu_aligned.sh                      # all 100 triplets
#   ./wikitext_unlearn_tofu_aligned.sh --only 002,012       # subset
#   ./wikitext_unlearn_tofu_aligned.sh --triplet triplet_001 --single   # single
#   ./wikitext_unlearn_tofu_aligned.sh --bs 8 --gas 8       # if OOM at BS=16
#   ./wikitext_unlearn_tofu_aligned.sh --dry-run
#
# Output:
#   saves/wikitext_unlearn_tofu/<TASK>/        per-triplet checkpoint
#   logs_wikitext_unlearn_tofu/<TASK>.log
#   logs_wikitext_unlearn_tofu/_summary.csv

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_UNLEARN_DIR="/media/volume/llm/unlearning/2.train-unlearn/open-unlearning"
TRIPLET_ROOT="/media/volume/llm/unlearning/1.data-preparation/data/wikitext_hdbscan_triplets"

SAVE_DIR="${SCRIPT_DIR}/saves/wikitext_unlearn_tofu"
LOG_DIR="${SCRIPT_DIR}/logs_wikitext_unlearn_tofu"
SUMMARY_CSV="${LOG_DIR}/_summary.csv"

GPU="0"
MODEL="Llama-3.1-8B-Instruct"
TRAINER="GradAscent"

# TOFU-aligned defaults — do NOT change without updating the fingerprint comment above.
BS=16             # per_device_train_batch_size (TOFU forget01 value)
GAS=4             # gradient_accumulation_steps  (TOFU forget01 value)
                  # effective_batch = BS * GAS * num_devices = 16 * 4 * 1 = 64
EPOCHS=5          # TOFU forget01 num_epochs
LR="1e-5"         # TOFU forget01 default
MAX_STEPS=3       # = epochs * forget_size / effective_batch = 5 * 50 / 64 ≈ 3
LR_SCHEDULER="linear"
WARMUP_STEPS=1    # TOFU: max(1, steps_per_epoch)
OPTIM="paged_adamw_32bit"   # TOFU optimizer
WEIGHT_DECAY="0.01"          # TOFU weight_decay
BF16=True         # TOFU: bf16=True

ATTN_IMPL="sdpa"
TRIPLET=""
SINGLE=0
ONLY=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)             GPU="$2";             shift 2 ;;
        --model)           MODEL="$2";           shift 2 ;;
        --trainer)         TRAINER="$2";         shift 2 ;;
        --bs)              BS="$2";              shift 2 ;;
        --gas)             GAS="$2";             shift 2 ;;
        --epochs)          EPOCHS="$2";          shift 2 ;;
        --lr)              LR="$2";              shift 2 ;;
        --max_steps)       MAX_STEPS="$2";       shift 2 ;;
        --warmup_steps)    WARMUP_STEPS="$2";    shift 2 ;;
        --optim)           OPTIM="$2";           shift 2 ;;
        --weight_decay)    WEIGHT_DECAY="$2";    shift 2 ;;
        --bf16)            BF16="$2";            shift 2 ;;
        --attn_impl)       ATTN_IMPL="$2";       shift 2 ;;
        --triplet)         TRIPLET="$2";         shift 2 ;;
        --single)          SINGLE=1;             shift   ;;
        --only)            ONLY="$2";            shift 2 ;;
        --dry-run)         DRY_RUN=1;            shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
    echo "triplet,status,wall_seconds,global_step,train_loss_final,task_name" > "${SUMMARY_CSV}"
fi

# Build triplet list
if [[ "${SINGLE}" == "1" ]]; then
    if [[ -z "${TRIPLET}" ]]; then
        echo "ERROR: --single requires --triplet"; exit 1
    fi
    ALL_TRIPLETS=("${TRIPLET}")
else
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
fi

echo "============================================"
echo "TOFU-aligned GradAscent unlearning"
echo "  Fingerprint:"
echo "    epoch=${EPOCHS}  BS×GAS=${BS}×${GAS}=$(( BS * GAS ))  max_steps=${MAX_STEPS}"
echo "    lr=${LR}  scheduler=${LR_SCHEDULER}  warmup_steps=${WARMUP_STEPS}"
echo "    optim=${OPTIM}  weight_decay=${WEIGHT_DECAY}  bf16=${BF16}"
echo "  GPU:          ${GPU}"
echo "  Model:        ${MODEL}"
echo "  Trainer:      ${TRAINER}"
echo "  Candidates:   ${#ALL_TRIPLETS[@]} triplets"
echo "  SAVE_DIR:     ${SAVE_DIR}"
echo "  LOG_DIR:      ${LOG_DIR}"
echo "============================================"

N_SKIP=0; N_RUN=0; N_FAIL=0
for TRIPLET in "${ALL_TRIPLETS[@]}"; do
    TASK_NAME="wikitext_${MODEL}_${TRIPLET}_${TRAINER}_tofu"
    OUTPUT_DIR="${SAVE_DIR}/${TASK_NAME}"
    FORGET_FILE="${TRIPLET_ROOT}/${TRIPLET}/train.json"
    RETAIN_FILE="${TRIPLET_ROOT}/${TRIPLET}/validation.json"

    if [[ ! -f "${FORGET_FILE}" || ! -f "${RETAIN_FILE}" ]]; then
        echo "[SKIP-missing] ${TRIPLET}"
        echo "${TRIPLET},missing,0,,,${TASK_NAME}" >> "${SUMMARY_CSV}"
        N_SKIP=$((N_SKIP+1)); continue
    fi

    if [[ -d "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
        echo "[SKIP-done]    ${TRIPLET}"
        echo "${TRIPLET},skipped,0,,,${TASK_NAME}" >> "${SUMMARY_CSV}"
        N_SKIP=$((N_SKIP+1)); continue
    fi

    echo "[RUN]          ${TRIPLET}"
    if [[ "${DRY_RUN}" == "1" ]]; then continue; fi

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
            ++trainer.args.max_steps="${MAX_STEPS}" \
            trainer.args.learning_rate="${LR}" \
            ++trainer.args.lr_scheduler_type="${LR_SCHEDULER}" \
            ++trainer.args.warmup_steps="${WARMUP_STEPS}" \
            trainer.args.optim="${OPTIM}" \
            trainer.args.weight_decay="${WEIGHT_DECAY}" \
            trainer.args.bf16="${BF16}" \
            trainer.args.logging_steps=1
    ) > "${TRAIN_LOG}" 2>&1
    RC=$?
    DUR=$(( $(date +%s) - T0 ))

    # Parse global_step and final train_loss from trainer_state.json
    GS=""; LOSS=""
    TSTATE="${OUTPUT_DIR}/trainer_state.json"
    if [[ -f "${TSTATE}" ]]; then
        GS=$(python3 -c "import json; d=json.load(open('${TSTATE}')); print(d.get('global_step',''))" 2>/dev/null || echo "")
        LOSS=$(python3 -c "import json; d=json.load(open('${TSTATE}')); lh=d.get('log_history',[]); print(lh[-1].get('train_loss','') if lh else '')" 2>/dev/null || echo "")
    fi

    if [[ ${RC} -eq 0 ]]; then
        echo "  → ok (${DUR}s, global_step=${GS}, train_loss=${LOSS})"
        echo "${TRIPLET},ok,${DUR},${GS},${LOSS},${TASK_NAME}" >> "${SUMMARY_CSV}"
        N_RUN=$((N_RUN+1))
    else
        echo "  → FAIL rc=${RC} (see ${TRAIN_LOG})"
        echo "${TRIPLET},fail_rc${RC},${DUR},${GS},${LOSS},${TASK_NAME}" >> "${SUMMARY_CSV}"
        N_FAIL=$((N_FAIL+1))
    fi
done

echo "============================================"
echo "Done:  run=${N_RUN}  skipped=${N_SKIP}  failed=${N_FAIL}"
echo "Summary: ${SUMMARY_CSV}"
echo "============================================"
