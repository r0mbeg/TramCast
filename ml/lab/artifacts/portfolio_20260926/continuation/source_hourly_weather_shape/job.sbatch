#!/bin/bash
#SBATCH --job-name=tram-portfolio-hourly_weather_shape
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-hourly_weather_shape-%j.log

set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_portfolio_hourly_weather_shape.sbatch worker
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-hourly_weather_shape-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.portfolio_conditional_shape --history artifacts/hourly_clean.csv \
    --output "$out/hourly_weather_shape" --weather "$out/external/weather_2025.csv" --hourly-weather "$out/external/weather_hourly_2025_era5.json"
