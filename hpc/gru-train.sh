#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train.out
#BSUB -e hpc/logs/train.err

# Submit your training run to a GPU queue:   bsub < hpc/train.sh
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
#BSUB -u s253065@dtu.dk
#BSUB -B
#BSUB -N

#set -eu
#cd "$(dirname "$0")/.."
set -e
source hpc/env.sh   # uv on the PATH, caches off the home quota

# Your training command, e.g.:
# uv run python train.py

mkdir -p results
: > results/gru_models-trained.txt

while read -r size layers rate epochs batch decay; do
    uv run python -m protgen.train_scripts.gru_train \
        --epochs "$epochs" --num-gru-layers "$layers" --gru-size "$size" \
        --learning-rate "$rate" --batch-size "$batch" --weight-decay "$decay"

    rate_label="${rate#0.}"
    echo "gru_h${size}_l${layers}_r${rate_label}_e${epochs}_b${batch}_d${decay}" >> results/gru_models-trained.txt
done <<'SETTINGS'
300 4 0.001 70 64 0.0
300 4 0.001 70 32 0.0
200 4 0.001 70 64 0.0001
300 4 0.001 70 64 0.0001
300 3 0.001 70 64 0.0
SETTINGS
