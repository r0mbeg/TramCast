#!/bin/bash
#SBATCH --job-name=tram-portfolio-ridge
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/portfolio_20260926/slurm-ridge-%j.log

set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores bash jobs/zhores_portfolio_ridge.sbatch worker
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-ridge-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.portfolio_experiment \
    --history artifacts/hourly_clean.csv --output "$out/ridge_probe" --families ridge
