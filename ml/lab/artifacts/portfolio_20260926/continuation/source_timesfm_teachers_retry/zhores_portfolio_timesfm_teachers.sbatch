#!/bin/bash
#SBATCH --job-name=tram-portfolio-timesfm-teachers
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=3G
#SBATCH --time=00:05:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-timesfm-teachers-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_portfolio_timesfm_teachers.sbatch worker "${1:-study}"
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/deps:$PWD/timesfm_deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
set -- "${2:-study}"
case "$1" in
    pilot) set -- --pilot ;;
    study) set -- ;;
    *) exit 2 ;;
esac
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-timesfm-teachers-$SLURM_JOB_ID.txt"
"$previous/.venv/bin/python" -m experiments.portfolio_timesfm --history artifacts/hourly_clean.csv \
    --model-path "$PWD/model-timesfm2p5" --output "$out/timesfm_teachers" --verified-july --teachers "$@"
