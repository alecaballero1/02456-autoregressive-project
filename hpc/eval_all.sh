#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

while read -r GRU_MODEL_NAME; do
    export GRU_MODEL_NAME
    bsub < hpc/eval.sh
done < results/gru_models.txt
