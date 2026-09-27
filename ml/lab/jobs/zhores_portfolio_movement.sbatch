#!/bin/bash
#SBATCH --job-name=tram-portfolio-move
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/portfolio_20260926/slurm-movement-%j.log

set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_portfolio_movement.sbatch worker
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
unset PYTHONPATH
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-movement-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.portfolio_movement --history artifacts/hourly_clean.csv \
    --selection "$out/ridge_probe/ridge/selection.json" --output "$out/movement"
