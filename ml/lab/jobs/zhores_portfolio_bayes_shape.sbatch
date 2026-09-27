#!/bin/bash
#SBATCH --job-name=tram-portfolio-bayes-shape
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-bayes-shape-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    if [ "${1:-}" = pilot ]; then
        exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_portfolio_bayes_shape.sbatch worker pilot
    fi
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_portfolio_bayes_shape.sbatch worker
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-bayes-shape-$SLURM_JOB_ID.txt"
if [ "${2:-}" = pilot ]; then
    set -- --pilot
else
    set --
fi
"$previous/.venv/bin/python" -m experiments.portfolio_bayes_shape --history artifacts/hourly_clean.csv --output "$out/bayes_shape" "$@"
