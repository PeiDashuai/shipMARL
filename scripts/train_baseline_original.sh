#!/bin/bash
# ============================================================================
# train_baseline_original.sh
# Purpose: Test training WITHOUT early stopping and stability controls
# This restores the original training configuration to verify if those
# mechanisms are causing the training failure
# ============================================================================

set -e

# Configuration
OUTPUT_DIR="TRAIN_baseline_original"
AIS_CONFIG="ais_comms/ais_config_baseline_v1.yaml"
ITERATIONS=200
MODEL="gnn_lstm"

# Training hyperparameters (same as original)
LR=1e-4                    # Lower learning rate for stability
ENTROPY_COEFF=0.015        # Slightly higher entropy for exploration
CLIP_PARAM=0.2             # Standard clip
TRAIN_BATCH=4000           # Standard batch size
SGD_MINIBATCH=256          # Standard minibatch
NUM_SGD_ITER=10            # Standard SGD iterations
NUM_WORKERS=4              # Parallel workers
ENVS_PER_WORKER=2          # Envs per worker

# Environment settings
N_SHIPS=2                  # Start with 2 ships
DT=0.5                     # Simulation timestep
T_MAX=180.0                # Max episode time

echo "============================================================================"
echo "Training Baseline Model (ORIGINAL - NO EARLY STOP)"
echo "============================================================================"
echo "Output:     ${OUTPUT_DIR}"
echo "AIS Config: ${AIS_CONFIG}"
echo "Iterations: ${ITERATIONS}"
echo "Model:      ${MODEL}"
echo "LR:         ${LR}"
echo "Lagrangian: ENABLED"
echo "Early Stop: DISABLED"
echo "Stability:  DISABLED"
echo "============================================================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run training WITHOUT early-stop and stability args
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
    --out-dir ${OUTPUT_DIR}

echo "============================================================================"
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints"
echo "Best model: ${OUTPUT_DIR}/best"
echo "============================================================================"
