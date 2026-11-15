#!/bin/sh
#BSUB -q gpuv100
#BSUB -J eval
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/eval_%J.out
#BSUB -e hpc/logs/eval_%J.err

# Score sample sets on a GPU, structures included:   bsub < hpc/eval.sh
#
# Run `bash hpc/setup_env.sh` once before this, to install the environment and download the
# ESMFold weights. Add your own sample CSVs to the list below; a name with no CSV is skipped with
# a message rather than failing the job, so this can be submitted before every model has been
# sampled. Each set lands in results/eval/<name>/.
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
#BSUB -u s214946@dtu.dk
#BSUB -B
#BSUB -N

set -eu
#cd "$(dirname "$0")/.."

cd /dtu/blackhole/0a/168650/02456-autoregressive-project
echo "Working directory: $PWD"

# SETS="control_real control_shuffled mock_data"
SETS="2"

for NAME in $SETS; do
    SAMPLES="results/sequences/$NAME/lstm_samples.csv"
    if [ ! -f "$SAMPLES" ]; then
        echo "== $NAME: no $SAMPLES, skipping"
        echo "available files:"
        find results/sequences -type f
        continue
    fi
    echo "== $NAME"
    bash hpc/run_eval.sh "$SAMPLES" "results/eval/2
done

echo "Done."
