#!/bin/bash
#SBATCH --job-name=tram-tirex-download
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-tirex-download-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores bash jobs/zhores_portfolio_tirex_download.sbatch worker
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="$PWD/deps:$PWD/tirex_deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-tirex-download-$SLURM_JOB_ID.txt"
sha256sum --check <<'HASH'
e470b2e1a4ad2fe6ab95012da6b92ffa20f69ff3a5d44b5b37ba3cd86ff2b2e0  tirex_package/tirex_ts-1.4.2-py3-none-any.whl
HASH
"$previous/.venv/bin/python" -m pip install --no-deps --target "$PWD/tirex_deps" tirex_package/tirex_ts-1.4.2-py3-none-any.whl
"$previous/.venv/bin/python" -m experiments.portfolio_tirex --history artifacts/hourly_clean.csv --model-path "$PWD/model-tirex" --output "$out/tirex" --download
