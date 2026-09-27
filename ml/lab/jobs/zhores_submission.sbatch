#!/bin/bash
#SBATCH --job-name=tram-submission
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/slurm-%j.log

set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores bash ./jobs/zhores_submission.sbatch worker
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=4 PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1
unset PYTHONPATH
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
scontrol show job "$SLURM_JOB_ID" > artifacts/slurm-allocation.txt
"$previous/.venv/bin/python" -m tests.test_pipeline
"$previous/.venv/bin/python" -m tests.test_chronos_experiment
"$previous/.venv/bin/python" -m experiments.chronos_experiment --history "$previous/artifacts/hourly_clean.csv" \
    --model-path "$previous/model" --output artifacts/results --submission
