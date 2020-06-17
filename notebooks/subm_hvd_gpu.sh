#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --partition gpu_big
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:0:00
#SBATCH --output logs/job_-%J.log

source /trinity/home/r.schutski/asr_speedup/venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/trinity/home/r.schutski/asr_speedup/venv/lib:/trinity/home/r.schutski/asr_speedup/venv/lib/python3.7

module load mpi/openmpi-3.1.2
module load gpu/cuda-10.1
echo `pwd`
cd /trinity/home/r.schutski/asr_speedup/Automatic-Speech-Recognition/notebooks

echo 'Running script:'
echo ${@}
#nodelist=`python get_nodelist.py --nl $SLURM_NODELIST --pn 4`
#echo 'Will run on nodes:'
#echo $nodelist
# horovodrun -np 16 -H $nodelist ${@}
# -H $nodelist \
mpirun -np 32 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 ${@}
