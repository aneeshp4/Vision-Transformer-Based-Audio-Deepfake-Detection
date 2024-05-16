#!/bin/bash

# Name of the job
#SBATCH --job-name=cs78_final_project

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Request the GPU partition
#SBATCH --partition gpuq

#SBATCH --mem=16G

# Request the GPU resources
#SBATCH --gres=gpu:k80:3

# Walltime (job duration)
#SBATCH --time=00:15:00

# Email notifications when the job starts and ends and fails
#SBATCH --mail-type=BEGIN,END, FAIL


#SBATCH --output=/dartfs-hpc/rc/home/g/f0055vg/Vision-Transformer-Based-Audio-Deepfake-Detection/output/test.out
#SBATCH --error=/dartfs-hpc/rc/home/g/f0055vg/Vision-Transformer-Based-Audio-Deepfake-Detection/error/test.err

nvidia-smi
echo $CUDA_VISIBLE_DEVICES
hostname
echo
echo
source /dartfs-hpc/rc/home/g/f0055vg/.bashrc
conda activate asit
cd /dartfs-hpc/rc/home/g/f0055vg/Vision-Transformer-Based-Audio-Deepfake-Detection/
python -u finetune.py