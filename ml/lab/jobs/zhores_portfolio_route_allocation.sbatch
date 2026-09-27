#!/bin/bash
#SBATCH --job-name=tram-portfolio-route-allocation
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:02:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-route-allocation-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores bash jobs/zhores_portfolio_route_allocation.sbatch worker "${1:-study}"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-route-allocation-$SLURM_JOB_ID.txt"
case "${2:-study}" in
    pilot) set -- --pilot ;;
    study) set -- ;;
    *) exit 2 ;;
esac
"$previous/.venv/bin/python" -m experiments.portfolio_route_allocation --history artifacts/hourly_clean.csv --output "$out/route_allocation" "$@"
