#!/bin/bash
#
# ===== SLURM Job Settings =====
#SBATCH --job-name=mcnp             # Job name
#SBATCH --partition=debug           # Partition
#SBATCH --ntasks=1                  # Single process
#SBATCH --cpus-per-task=11          # Number of OpenMP threads
#SBATCH --time=0                    # Max runtime
#SBATCH --output=slurm-%j.out       # Stdout
#SBATCH --error=slurm-%j.err        # Stderr

# ===== Environment Setup =====
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Export MCNP input files or other variables
export DATAPATH="/home/camden/Data/MCNP/MCNP_DATA"

EXE="/home/camden/MyPrograms/MCNP-thermal-otf/MCNP_6.3_SOURCE/mcnp-src/mcnp-6.3.0-Source/mcnp6/build/mcnp6"
INP="model.mcnp"
OTF="/home/camden/Data/OTF/Test_ENDF8_0_OTF_Files/tsl-HinH2O_OTF.h5"

# ===== Run MCNP =====
$EXE inp=$INP otf $OTF tasks $OMP_NUM_THREADS

