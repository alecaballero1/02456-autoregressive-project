#!/bin/bash
#BSUB -q gpuv100
#BSUB -J sample
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 20:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/sample_%J.out
#BSUB -e hpc/logs/sample_%J.err

# Submit your sampling run to a GPU queue:   bsub < hpc/sample.sh
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
#BSUB -u s214946@dtu.dk
#BSUB -B
#BSUB -N

set -eu

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

#cd "$(dirname "$0")/.."
source "hpc/env.sh"   # uv on the PATH, caches off the home quota
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Your training command, e.g.:
uv run python -m protgen.train_scripts.lstm_sample