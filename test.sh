#!/bin/bash
#SBATCH --job-name=cv-project        # Job name
#SBATCH --output=log%j.log       # Both stdout and stderr go here
#SBATCH --error=log%j.log        # Optional, can be the same file
#SBATCH --time=01:00:00             # Max runtime
#SBATCH --partition=ENSTA-l40s      # Partition
#SBATCH --gpus=1                    # Number of GPUs
#SBATCH --cpus-per-task=4           # CPUs
#SBATCH --mem=16G                   # Memory

# Activate virtual environment
nvidia-smi
source $HOME/dl_env/bin/activate

cd $HOME/CV-project-OOD
# Run Python script and merge stderr into stdout

python plot_visualizations.py 2>&1