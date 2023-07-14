#!/bin/bash
## Running code on SLURM cluster
## https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html
#SBATCH -J anad
#SBATCH -o anad-%j.out
#SBATCH -e anad-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

singularity exec -e --pwd /app -B $(pwd)/anad-logs:/app/examples/cardiovascular-risk --nv docker://spartan300/anad python evolve-autoencoder.py