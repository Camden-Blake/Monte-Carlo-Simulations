#!/bin/bash
#
# ===== SLURM Job Settings =====
#SBATCH --job-name=openmc           # Job name
#SBATCH --partition=debug           # Partition
#SBATCH --ntasks=1                  # Single process
#SBATCH --cpus-per-task=11          # Number of OpenMP threads
#SBATCH --time=0                    # Max runtime
#SBATCH --output=slurm-%j.out       # Stdout
#SBATCH --error=slurm-%j.err        # Stderr

# ===== Environment Setup =====
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Export OpenMC input files or other variables
export OPENMC_CROSS_SECTIONS="/home/camden/Data/OpenMC/endfb-viii.0-hdf5/cross_sections.xml"

EXE="/home/camden/MyPrograms/openmc-thermal-otf/build/bin/openmc"

# ===== Run OpenMC =====
$EXE

