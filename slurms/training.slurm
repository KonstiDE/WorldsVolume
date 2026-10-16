#!/bin/bash
#SBATCH --job-name=VolumeDL                 # Job name
#SBATCH --output=output.txt                 # Standard output file
#SBATCH --error=error.txt                   # Standard error file
#SBATCH --clusters=hpda2                    # hpda1 for jupyter, hpda2 for rest
#SBATCH --partition=hpda2_testgpu           # Partition or queue name
#SBATCH --gres=gpu:1                        # Amount of GPUs
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --cpus-per-task=16                  # Number of CPU cores per task
#SBATCH --mem=512gb                         # Max memory
#SBATCH --time=0-02:00:00                   # Maximum runtime (D-HH:MM:SS)

#SBATCH --exclude=hpdar04c01s02,hpdar04c01s01

#Load necessary modules (if needed)
module load micromamba

#Your job commands go here
#For example:
micromamba run -p /dss/dsshome1/0C/di97nev/micromamba/envs/python python /dss/dsshome1/0C/di97nev/WorldsVolume/training.py