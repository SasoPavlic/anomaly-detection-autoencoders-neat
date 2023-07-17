#!/bin/bash
## Running code on SLURM cluster
#SBATCH -J anad-three
#SBATCH -o anad-three-100gen-%j.out
#SBATCH -e anad-three-100gen-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=8GB  # memory per GPU
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

#singularity exec -e --pwd /app/examples/cardiovascular-risk -B $(pwd)/logs:/app/examples/cardiovascular-risk/logs,$(pwd)/config:/app/examples/cardiovascular-risk/config --nv docker://spartan300/anad:latest python evolve-autoencoder.py | tee $output_file

singularity exec -e \
    --pwd /app/examples/cardiovascular-risk \
    -B $(pwd)/logs:/app/examples/cardiovascular-risk/logs,$(pwd)/config:/app/examples/cardiovascular-risk/config \
    --nv docker://spartan300/anad:latest \
    python evolve-autoencoder.py | tee $output_file