#!/bin/bash
#SBATCH --job-name=tram-portfolio-verified-timesfm-blend
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-verified-timesfm-blend-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_portfolio_verified_timesfm_blend.sbatch worker
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-verified-timesfm-blend-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.portfolio_direct_blend --history artifacts/hourly_clean.csv \
    --output "$out/verified_timesfm_blend" --timesfm --verified-july
