#!/bin/bash
#SBATCH --job-name=VAE_MOL
#SBATCH --output=salida_data_prep.out
#SBATCH --error=error_data_prep.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00

/etc/profile

source /home/cperez/miniconda3/bin/activate vae-mol

python Tesis_VAE/src/data_prep.py