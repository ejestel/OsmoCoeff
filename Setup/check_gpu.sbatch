#!/bin/bash
# One-GPU test job for the OsmoCoeff environment. Submit from the repository root:
#
#   sbatch Setup/check_gpu.sbatch
#
# Options given to sbatch override the #SBATCH lines below (e.g. --qos, --partition).
# For another environment location:  OSMOCOEFF_ENV=/path/to/env sbatch Setup/check_gpu.sbatch
#SBATCH --job-name=check_gpu
#SBATCH --partition=blanca-shirts
#SBATCH --qos=blanca-shirts
#SBATCH --account=blanca-shirts
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=check_gpu_%j.log

module load anaconda
export PYTHONNOUSERSITE=1
conda activate "${OSMOCOEFF_ENV:-/projects/$USER/software/anaconda/env/polymerist-osmocoeff}" || exit 1

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi -L

python Setup/check_environment.py --require-gpu
