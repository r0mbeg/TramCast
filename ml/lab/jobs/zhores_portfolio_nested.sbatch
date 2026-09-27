#!/bin/bash
#SBATCH --job-name=tram-portfolio-inner
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/portfolio_20260926/slurm-inner-%j.log

set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores bash jobs/zhores_portfolio_nested.sbatch worker
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1
unset PYTHONPATH
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-inner-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.portfolio_chronos --nested \
    --history artifacts/hourly_clean.csv --model-path "$previous/model" --output "$out/inner_gpu"
