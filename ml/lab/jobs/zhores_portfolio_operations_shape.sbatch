#!/bin/bash
#SBATCH --job-name=tram-portfolio-operations_shape
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-operations_shape-%j.log

set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores bash jobs/zhores_portfolio_operations_shape.sbatch worker
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-operations_shape-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.portfolio_conditional_shape --history artifacts/hourly_clean.csv \
    --output "$out/operations_shape" --weather "$out/external/weather_2025.csv" --operations
