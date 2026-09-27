#!/bin/bash
#SBATCH --job-name=tram-dense-teachers
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:10:00
#SBATCH --output=dense_student/slurm-%j.log
set -euo pipefail
if [ "${1:-}" != worker ]; then
    exec srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores bash dense_student/teacher.sbatch worker "$@"
fi
shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$PWD/tabpfn_deps:$PWD/deps:$PWD/timesfm_deps:$PWD"
/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926/.venv/bin/python dense_student/teacher.py "$@"
