#!/bin/bash
#SBATCH --job-name=tram-portfolio-robust-annual
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-robust-annual-%j.log
set -euo pipefail
mode=${1:-pilot}
if [ "$mode" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores bash jobs/zhores_portfolio_robust_annual.sbatch worker "$mode"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-robust-annual-$SLURM_JOB_ID.txt"
args=()
if [ "${2:-}" = pilot ]; then args+=(--pilot); fi
"$previous/.venv/bin/python" -m experiments.portfolio_robust_annual --history artifacts/hourly_clean.csv --output "$out/robust_annual" "${args[@]}"
