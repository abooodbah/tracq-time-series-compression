#!/bin/bash
#SBATCH -J tracq-scale
#SBATCH -o tracq-scale.%j.out
#SBATCH -e tracq-scale.%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 01:45:00

set -e
cd $SCRATCH/tracq-tb
source $SCRATCH/tracq-tb/venv/bin/activate
echo "node: $(hostname), cores: ${SLURM_CPUS_PER_TASK}"
python3 scripts/scaling_study.py $SCRATCH/tracq-tb/scaling_result.json
