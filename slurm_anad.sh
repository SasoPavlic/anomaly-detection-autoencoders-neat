#!/bin/bash
## Running code on SLURM cluster
#SBATCH -J three
#SBATCH -o three-%j.out
#SBATCH -e three-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=8GB  # memory per GPU
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

singularity exec -e \
    --pwd /app/examples/cardiovascular-risk \
    -B $(pwd)/logs:/app/examples/cardiovascular-risk/logs,$(pwd)/config:/app/examples/cardiovascular-risk/config \
    --nv docker://spartan300/anad:latest \
    python evolve-autoencoder.py -cl three | tee $output_file