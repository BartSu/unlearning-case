#!/usr/bin/env bash
set -euo pipefail

cd /media/volume/llm/unlearning/2.extract-ppl

TRIPLETS="triplet_001,triplet_002,triplet_003,triplet_004,triplet_005,triplet_011,triplet_012,triplet_013,triplet_014,triplet_015,triplet_021,triplet_022,triplet_023,triplet_024,triplet_025,triplet_031,triplet_032,triplet_033,triplet_034,triplet_035,triplet_041,triplet_042,triplet_043,triplet_044,triplet_045,triplet_051,triplet_052,triplet_053,triplet_054,triplet_055,triplet_061,triplet_062,triplet_063,triplet_064,triplet_065,triplet_071,triplet_072,triplet_073,triplet_074,triplet_075,triplet_081,triplet_082,triplet_083,triplet_084,triplet_085,triplet_091,triplet_092,triplet_093,triplet_094,triplet_095"

DATA_DIR=/media/volume/llm/unlearning/1.data-preparation/data/wikitext_hdbscan_triplets
SAVES_DIR=/media/volume/llm/unlearning/1.data-preparation/unlearn/saves/wikitext_unlearn
LOGDIR=/media/volume/llm/unlearning/2.extract-ppl/logs_n50
mkdir -p "$LOGDIR"

BATCH_SIZE=16

echo "[$(date -Is)] START baseline (50 triplets, resume, bs=$BATCH_SIZE)" | tee -a "$LOGDIR/run.log"
python eval_wikitext_perplexity.py \
    --baseline \
    --data_dir "$DATA_DIR" \
    --triplets "$TRIPLETS" \
    --batch_size "$BATCH_SIZE" \
    --resume \
    2>&1 | tee -a "$LOGDIR/baseline.log"

echo "[$(date -Is)] START cross-eval (50x50, resume, bs=$BATCH_SIZE)" | tee -a "$LOGDIR/run.log"
python eval_wikitext_perplexity.py \
    --saves_dir "$SAVES_DIR" \
    --data_dir "$DATA_DIR" \
    --triplets "$TRIPLETS" \
    --batch_size "$BATCH_SIZE" \
    --resume \
    2>&1 | tee -a "$LOGDIR/cross.log"

echo "[$(date -Is)] DONE" | tee -a "$LOGDIR/run.log"
touch "$LOGDIR/DONE"
