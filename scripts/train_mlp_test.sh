#!/bin/bash
# Test with MLP model (no GNN) to isolate the issue
set -e

OUTPUT_DIR="TRAIN_mlp_test"
ITERATIONS=100
MODEL="mlp"  # Use MLP instead of GNN

echo "============================================================================"
echo "Training with MLP Model (No GNN)"
echo "============================================================================"

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=. python scripts/train_rllib_ppo_pf.py \
    --model ${MODEL} \
    --iterations ${ITERATIONS} \
    --use-lagrangian \
    --N 2 \
    --dt 0.5 \
    --T_max 180.0 \
    --lr 3e-4 \
    --clip-param 0.2 \
    --entropy-coeff 0.01 \
    --train-batch-size 4000 \
    --sgd-minibatch-size 256 \
    --num-sgd-iter 10 \
    --num-workers 4 \
    --num-envs-per-worker 2 \
    --checkpoint-freq 10 \
    --out-dir ${OUTPUT_DIR}

echo "Training complete!"
