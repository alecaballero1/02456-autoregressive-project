#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

EPOCHS_VALUES=(60)
LAYERS_VALUES=(1 2)
GRU_SIZE_VALUES=(50 100 200)
LEARNING_RATE_VALUES=(0.0003 0.001)
BATCH_SIZE_VALUES=(64)
WEIGHT_DECAY_VALUES=(0.0 0.0001)

mkdir -p results hpc/logs
: > results/gru_models.txt

for epochs in "${EPOCHS_VALUES[@]}"; do
for layers in "${LAYERS_VALUES[@]}"; do
for size in "${GRU_SIZE_VALUES[@]}"; do
for rate in "${LEARNING_RATE_VALUES[@]}"; do
for batch in "${BATCH_SIZE_VALUES[@]}"; do
for decay in "${WEIGHT_DECAY_VALUES[@]}"; do
    export GRU_TRAIN_ARGS="--epochs $epochs --num-gru-layers $layers --gru-size $size --learning-rate $rate --batch-size $batch --weight-decay $decay"
    bsub < hpc/train.sh

    rate_label="${rate#0.}"
    while [[ "$rate_label" == *0 ]]; do
        rate_label="${rate_label%0}"
    done
    echo "gru_h${size}_l${layers}_r${rate_label}_e${epochs}_b${batch}_d${decay}" >> results/gru_models.txt
done
done
done
done
done
done
