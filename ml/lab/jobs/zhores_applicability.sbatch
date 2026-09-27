#!/bin/bash
#SBATCH --job-name=tram-applicability-M1
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/methodology_20260927/route/slurm-M1-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_applicability.sbatch worker
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/methodology_20260927/route
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-M1-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.applicability gpu --history artifacts/hourly_clean.csv \
    --model-path "$previous/model" --output "$out/m1"
