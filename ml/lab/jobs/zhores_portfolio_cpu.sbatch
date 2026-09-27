#!/bin/bash
#SBATCH --job-name=tram-portfolio-cpu
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/portfolio_20260926/slurm-cpu-%j.log

set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores bash jobs/zhores_portfolio_cpu.sbatch worker
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-cpu-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m tests.test_portfolio_experiment
"$previous/.venv/bin/python" -m experiments.portfolio_experiment \
    --history artifacts/hourly_clean.csv --output "$out/cpu"
