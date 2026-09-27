#!/bin/bash
#SBATCH --job-name=tram-applicability-M2
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=artifacts/methodology_20260927/route/slurm-M2-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash artifacts/methodology_20260927/route/m2.sbatch worker
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/methodology_20260927/route
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-M2-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.applicability cpu --history artifacts/hourly_clean.csv --output "$out/m2"
