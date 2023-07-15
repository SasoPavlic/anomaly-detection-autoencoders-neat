#!/bin/bash
## Running code on SLURM cluster
#SBATCH -J anad
#SBATCH -o anad-%j.out
#SBATCH -e anad-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=8GB  # memory per GPU
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

singularity exec -e --pwd /app/examples/cardiovascular-risk -B $(pwd)/logs:/app/examples/cardiovascular-risk/logs,$(pwd)/config:/app/examples/cardiovascular-risk/config --nv docker://spartan300/anad:latest python evolve-autoencoder.py