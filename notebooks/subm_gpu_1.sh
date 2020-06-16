#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition gpu_big
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:0:00
#SBATCH --output logs/job_-%J.log

source /trinity/home/r.schutski/asr_speedup/venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/trinity/home/r.schutski/asr_speedup/venv/lib:/trinity/home/r.schutski/asr_speedup/venv/lib/python3.7

module load gpu/cuda-10.1
echo `pwd`
cd /trinity/home/r.schutski/asr_speedup/Automatic-Speech-Recognition/notebooks

echo 'Running script:'
echo ${1}

python3 ${1}
