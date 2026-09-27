#!/bin/bash
#SBATCH --job-name=tram-portfolio-mix
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/portfolio_20260926/slurm-combine-%j.log

set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_portfolio_combine.sbatch worker
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-combine-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.portfolio_combine --history artifacts/hourly_clean.csv --root "$out"
