#!/bin/bash
#SBATCH --job-name=tram-runtime030-check
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1
portfolio=/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926
export PYTHONPATH="$PWD:$portfolio/tabpfn_deps:$portfolio/deps"
previous=/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv -lms 200 > gpu-memory.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
scontrol show job "$SLURM_JOB_ID" > allocation.txt
srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores "$previous/.venv/bin/python" verify.py
