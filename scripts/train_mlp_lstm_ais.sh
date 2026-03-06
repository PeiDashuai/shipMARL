#!/bin/bash
# ============================================================================
# train_mlp_lstm_ais.sh
# Purpose: Train MLP model WITH AIS observations enabled
# ============================================================================

set -e

OUTPUT_DIR="TRAIN_mlp_ais"
AIS_CONFIG="ais_comms/ais_config_baseline_v1.yaml"
ITERATIONS=500
MODEL="mlp"

# Training hyperparameters
LR=1e-4
ENTROPY_COEFF=0.015
CLIP_PARAM=0.2
TRAIN_BATCH=4000
SGD_MINIBATCH=256
NUM_SGD_ITER=10
NUM_WORKERS=4
ENVS_PER_WORKER=2

# Environment settings
N_SHIPS=2
DT=0.5
T_MAX=180.0

# Early stopping
EARLY_STOP=true
EARLY_STOP_SUCC=0.95
EARLY_STOP_PATIENCE=10
EARLY_STOP_MIN_ITER=80

# Stability controls
LR_DECAY_ON_PLATEAU=true
LR_DECAY_FACTOR=0.5
LR_DECAY_SUCC=0.85
LR_MIN=5e-5
STOP_ON_COLLAPSE=true
COLLAPSE_DROP=0.10

echo "============================================================================"
echo "Training MLP Model WITH AIS Observations"
echo "============================================================================"
echo "Output:     ${OUTPUT_DIR}"
echo "AIS Config: ${AIS_CONFIG}"
echo "Iterations: ${ITERATIONS} (max)"
echo "Model:      ${MODEL}"
echo "LR:         ${LR} (decay=${LR_DECAY_ON_PLATEAU}, min=${LR_MIN})"
echo "Clip:       ${CLIP_PARAM}"
echo "Lagrangian: ENABLED"
echo "AIS:        ENABLED"
echo "Early Stop: ${EARLY_STOP} (succ>=${EARLY_STOP_SUCC} for ${EARLY_STOP_PATIENCE} iters, min=${EARLY_STOP_MIN_ITER})"
echo "Collapse:   ${STOP_ON_COLLAPSE} (drop>=${COLLAPSE_DROP})"
echo "============================================================================"

mkdir -p "${OUTPUT_DIR}"

# Build early stop args
EARLY_STOP_ARGS=""
if [ "${EARLY_STOP}" = "true" ]; then
    EARLY_STOP_ARGS="--early-stop --early-stop-succ ${EARLY_STOP_SUCC} --early-stop-patience ${EARLY_STOP_PATIENCE} --early-stop-min-iter ${EARLY_STOP_MIN_ITER}"
fi

# Build stability args
STABILITY_ARGS=""
if [ "${LR_DECAY_ON_PLATEAU}" = "true" ]; then
    STABILITY_ARGS="${STABILITY_ARGS} --lr-decay-on-plateau --lr-decay-factor ${LR_DECAY_FACTOR} --lr-decay-succ-threshold ${LR_DECAY_SUCC} --lr-min ${LR_MIN}"
fi
if [ "${STOP_ON_COLLAPSE}" = "true" ]; then
    STABILITY_ARGS="${STABILITY_ARGS} --stop-on-collapse --collapse-drop-threshold ${COLLAPSE_DROP}"
fi

# Run training with MLP model and AIS enabled
PYTHONPATH=. python scripts/train_rllib_ppo_pf.py \
    --model ${MODEL} \
    --iterations ${ITERATIONS} \
    --use-ais-obs \
    --ais-cfg-path ${AIS_CONFIG} \
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
    --out-dir ${OUTPUT_DIR} \
    ${EARLY_STOP_ARGS} \
    ${STABILITY_ARGS}

echo "============================================================================"
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints"
echo "Best model: ${OUTPUT_DIR}/best"
echo "============================================================================"
