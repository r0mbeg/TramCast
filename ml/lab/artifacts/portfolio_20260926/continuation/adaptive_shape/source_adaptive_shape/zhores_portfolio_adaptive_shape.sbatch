#!/bin/bash
#SBATCH --job-name=tram-portfolio-adaptive-shape
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-adaptive-shape-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_portfolio_adaptive_shape.sbatch worker "${1:-pilot}"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/tabpfn_deps:$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-adaptive-shape-$SLURM_JOB_ID.txt"
case "${2:-pilot}" in
    pilot) set -- --pilot ;;
    study) set -- ;;
    *) exit 2 ;;
esac
"$previous/.venv/bin/python" -m experiments.portfolio_adaptive_shape --history artifacts/hourly_clean.csv --model-path "$PWD/model-tabpfn/tabpfn-v2-regressor.ckpt" --output "$out/adaptive_shape" "$@"
