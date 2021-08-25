#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --job-name=train_ae_mse_lr-3
#SBATCH --output=log_%x.out
#SBATCH --gres=gpu:TU104:1
#SBATCH --qos=longrunning

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR


echo "[SERVER]Starting slurm job"

python main.py

echo "[SERVER]Slurm job done!"