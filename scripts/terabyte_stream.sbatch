#!/bin/bash
#SBATCH -J tracq-tb
#SBATCH -o tracq-tb.%j.out
#SBATCH -e tracq-tb.%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 00:40:00

set -e
cd $SCRATCH/tracq-tb
source $SCRATCH/tracq-tb/venv/bin/activate
CORES=${SLURM_CPUS_PER_TASK}
echo "node: $(hostname), cores: $CORES"
python3 scripts/terabyte_stream.py 1000 $CORES $SCRATCH/tracq-tb/tb_result.json
