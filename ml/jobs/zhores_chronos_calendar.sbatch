#!/bin/bash
#SBATCH --job-name=tram-calendar
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2966
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/slurm-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores bash ./jobs/zhores_chronos_calendar.sbatch worker
fi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset PYTHONPATH
previous="$HOME/codex_runs/tram-chronos-20260926"
python_experiment="$previous/.venv/bin/python"
date -u
scontrol show job "$SLURM_JOB_ID" > artifacts/slurm-allocation.txt
"$python_experiment" -m pip freeze > artifacts/environment-freeze.txt
"$python_experiment" -m tests.test_chronos_experiment
"$python_experiment" -m tests.test_calendar_experiment
"$python_experiment" -m tests.test_chronos_calendar_experiment
"$python_experiment" -m experiments.chronos_calendar_experiment --model-path "$previous/model" \
    --reference "$previous/artifacts/results/predictions.csv" --output artifacts/results
date -u
