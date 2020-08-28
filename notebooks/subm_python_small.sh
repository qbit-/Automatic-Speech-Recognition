#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition gpu_small
#SBATCH --gres=gpu:1
#SBATCH --time=12:0:00
#SBATCH --output ./logs/job_-%J.log
#SBATCH --error ./logs/job-%J.errorlog

module unload python/python-3.6.8
source /trinity/home/g.leleitner/lab/Horovod/asr_3.7/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/data/home/g.leleitner/.local/lib
export PATH=$PATH:/gpfs/data/home/g.leleitner/.local/bin
module load gpu/cuda-10.1

echo 'Running script:'
echo ${@}
python3 ${@}
