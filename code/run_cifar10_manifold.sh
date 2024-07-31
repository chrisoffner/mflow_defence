#!/bin/bash
#SBATCH --job-name=python_gpu_test      # Job name
#SBATCH -G 1                            # Request one GPU
#SBATCH --account=cil                   # course tag
#SBATCH --time=24:00:00                 # Time limit hh:mm:ss

python ../notebooks/cifar10_manifold.py