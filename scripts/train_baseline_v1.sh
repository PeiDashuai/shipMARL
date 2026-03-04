#!/bin/bash
# ============================================================================
# train_baseline_v1.sh
# Purpose: Train a stable foundation model for collision avoidance with AIS
# Expected outcome: >90% success rate after 300-500 iterations
# ============================================================================

set -e

# Configuration
OUTPUT_DIR="TRAIN_baseline_v1"
AIS_CONFIG="ais_comms/ais_config_baseline_v1.yaml"
ITERATIONS=500
MODEL="gnn_lstm"

# Training hyperparameters (conservative for stability)
LR=1e-4                    # Lower learning rate for stability
ENTROPY_COEFF=0.015        # Slightly higher entropy for exploration
TRAIN_BATCH=4000           # Standard batch size
SGD_MINIBATCH=256          # Standard minibatch
NUM_SGD_ITER=10            # Standard SGD iterations
NUM_WORKERS=4              # Parallel workers
ENVS_PER_WORKER=2          # Envs per worker

# Environment settings
N_SHIPS=2                  # Start with 2 ships
DT=0.5                     # Simulation timestep
T_MAX=180.0                # Max episode time

# Early stopping (prevents policy collapse from overtraining)
EARLY_STOP=true            # Enable early stopping
EARLY_STOP_SUCC=0.95       # Stop when success rate >= 95%
EARLY_STOP_PATIENCE=20     # Require 20 consecutive iterations
EARLY_STOP_MIN_ITER=100    # Don't stop before iter 100

echo "============================================================================"
echo "Training Baseline Model v1"
echo "============================================================================"
echo "Output:     ${OUTPUT_DIR}"
echo "AIS Config: ${AIS_CONFIG}"
echo "Iterations: ${ITERATIONS} (max)"
echo "Model:      ${MODEL}"
echo "LR:         ${LR}"
echo "Lagrangian: ENABLED"
echo "Early Stop: ${EARLY_STOP} (succ>=${EARLY_STOP_SUCC} for ${EARLY_STOP_PATIENCE} iters, min=${EARLY_STOP_MIN_ITER})"
echo "============================================================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build early stop args
EARLY_STOP_ARGS=""
if [ "${EARLY_STOP}" = "true" ]; then
    EARLY_STOP_ARGS="--early-stop --early-stop-succ ${EARLY_STOP_SUCC} --early-stop-patience ${EARLY_STOP_PATIENCE} --early-stop-min-iter ${EARLY_STOP_MIN_ITER}"
fi

# Run training
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
    --entropy-coeff ${ENTROPY_COEFF} \
    --train-batch-size ${TRAIN_BATCH} \
    --sgd-minibatch-size ${SGD_MINIBATCH} \
    --num-sgd-iter ${NUM_SGD_ITER} \
    --num-workers ${NUM_WORKERS} \
    --num-envs-per-worker ${ENVS_PER_WORKER} \
    --checkpoint-freq 10 \
    --out-dir ${OUTPUT_DIR} \
    ${EARLY_STOP_ARGS}

echo "============================================================================"
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints"
echo "Best model: ${OUTPUT_DIR}/best"
echo "============================================================================"
