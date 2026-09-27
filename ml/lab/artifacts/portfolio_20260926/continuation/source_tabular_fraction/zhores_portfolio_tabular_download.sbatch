#!/bin/bash
#SBATCH --job-name=tram-tabular-download
#SBATCH --partition=ais-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=artifacts/portfolio_20260926/continuation/slurm-tabular-download-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores bash jobs/zhores_portfolio_tabular_download.sbatch worker
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONPATH="$PWD/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
out=artifacts/portfolio_20260926/continuation
scontrol show job "$SLURM_JOB_ID" > "$out/allocation-tabular-download-$SLURM_JOB_ID.txt"
sha256sum --check <<'HASH'
04e3bb989e9328d510ea4fccb6c6a36c8269630685d460af4df5eb268206bf21  tabpfn_package/tabpfn-2.0.9-py3-none-any.whl
HASH
"$previous/.venv/bin/python" -m experiments.portfolio_tabular_fraction --history artifacts/hourly_clean.csv --model-path model-tabpfn/tabpfn-v2-regressor.ckpt --output "$out/tabular_fraction" --download
"$previous/.venv/bin/python" -m pip install --no-deps --target "$PWD/tabpfn_deps" tabpfn_package/*.whl
