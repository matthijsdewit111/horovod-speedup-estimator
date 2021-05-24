#!/bin/bash
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=2

export OMP_NUM_THREADS=1

mpirun -np "$SLURM_NTASKS" python allreduce_timing.py lisa
