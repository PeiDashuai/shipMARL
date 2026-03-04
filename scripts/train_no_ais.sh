#!/bin/bash
# ============================================================================
# train_no_ais.sh
# Purpose: Test training WITHOUT AIS simulation (pure geometry observations)
# This tests if the base training works without AIS complexity
# ============================================================================

set -e

OUTPUT_DIR="TRAIN_no_ais"
ITERATIONS=100
MODEL="gnn_lstm"

LR=1e-4
ENTROPY_COEFF=0.015
CLIP_PARAM=0.2
TRAIN_BATCH=4000
SGD_MINIBATCH=256
NUM_SGD_ITER=10
NUM_WORKERS=4
ENVS_PER_WORKER=2

N_SHIPS=2
DT=0.5
T_MAX=180.0

echo "============================================================================"
echo "Training WITHOUT AIS (Pure Geometry Observations)"
echo "============================================================================"
echo "Output:     ${OUTPUT_DIR}"
echo "Iterations: ${ITERATIONS}"
echo "Model:      ${MODEL}"
echo "AIS:        DISABLED"
echo "============================================================================"

mkdir -p "${OUTPUT_DIR}"

# Run training WITHOUT --use-ais-obs flag
PYTHONPATH=. python scripts/train_rllib_ppo_pf.py \
    --model ${MODEL} \
    --iterations ${ITERATIONS} \
    --use-lagrangian \
    --N ${N_SHIPS} \
    --dt ${DT} \
    --T_max ${T_MAX} \
    --lr ${LR} \
    --clip-param ${CLIP_PARAM} \
    --entropy-coeff ${ENTROPY_COEFF} \
    --train-batch-size ${TRAIN_BATCH} \
    --sgd-minibatch-size ${SGD_MINIBATCH} \
    --num-sgd-iter ${NUM_SGD_ITER} \
    --num-workers ${NUM_WORKERS} \
    --num-envs-per-worker ${ENVS_PER_WORKER} \
    --checkpoint-freq 10 \
    --out-dir ${OUTPUT_DIR}

echo "============================================================================"
echo "Training complete!"
echo "============================================================================"
