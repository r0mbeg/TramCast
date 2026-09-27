#!/bin/bash
#SBATCH --job-name=tram-timesfm-download
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-timesfm-download-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash jobs/zhores_portfolio_timesfm_download.sbatch worker
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="$PWD/deps:$PWD/timesfm_deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-timesfm-download-$SLURM_JOB_ID.txt"
sha256sum --check <<'HASH'
c7bde94beb1651e1251cdf1e9d09cf6f015e0218d038a4a836a170dc70b08071  timesfm_package/timesfm-2.0.2-py3-none-any.whl
HASH
"$previous/.venv/bin/python" -m pip install --no-deps --target "$PWD/timesfm_deps" timesfm_package/timesfm-2.0.2-py3-none-any.whl
"$previous/.venv/bin/python" -m experiments.portfolio_timesfm --history artifacts/hourly_clean.csv \
    --model-path "$PWD/model-timesfm2p5" --output "$out/timesfm" --download
