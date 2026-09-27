#!/bin/bash
#SBATCH --job-name=tram-portfolio-seasonal-gate
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:02:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-seasonal-gate-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores bash jobs/zhores_portfolio_seasonal_gate.sbatch worker
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-seasonal-gate-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.portfolio_seasonal_gate --history artifacts/hourly_clean.csv --output "$out/seasonal_gate"
