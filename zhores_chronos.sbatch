#!/bin/bash
#SBATCH --job-name=tram-chronos
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2966
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/slurm-%j.log

set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores bash ./zhores_chronos.sbatch worker
fi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=8
export HF_HOME="$PWD/hf-cache" HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1
unset PYTHONPATH
date -u
scontrol show job "$SLURM_JOB_ID" > artifacts/slurm-allocation.txt
"$HOME/miniconda3/envs/py312/bin/python" -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'torch==2.6.0+cu124' 'numpy==1.26.4' 'pandas==2.2.3' \
    'chronos-forecasting==2.3.2' 'transformers==4.57.6'
.venv/bin/python -m pip freeze > artifacts/environment-freeze.txt
.venv/bin/python test_pipeline.py
.venv/bin/python test_chronos_experiment.py
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from chronos_experiment import REVISION
snapshot_download("amazon/chronos-2", revision=REVISION, local_dir="model", max_workers=4,
                  allow_patterns=["config.json", "model.safetensors"])
PY
.venv/bin/python chronos_experiment.py --model-path model --output artifacts/results
date -u
