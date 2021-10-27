#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

source $CONDA_BIN
conda activate TestEnv

echo "CONDA_BIN: $CONDA_BIN"
echo "HOSTNAME: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "GPU IDs: $SLURM_JOB_GPUS"
echo "PYTHON VERSION: $(python --version)"
echo "PATH: $PATH"
echo $(python -c "import torch; print('TORCH VERSION:', torch.__version__)")

echo "======================= EXAMPLE 1 ==========================="
python example1.py
echo "======================= EXAMPLE 2 ==========================="
python example2.py
echo "======================= EXAMPLE 3 ==========================="
python example3.py
