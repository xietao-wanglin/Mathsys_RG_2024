#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --array=0-19

module load GCCcore/11.3.0 Python/3.10.4

srun python ./Mathsys_RG_2024/RemoteLauncher.py $SLURM_ARRAY_TASK_ID