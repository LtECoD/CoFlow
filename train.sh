#! /bin/bash

#SBATCH --job-name train
#SBATCH -p PARTION
#SBATCH -N NODE_NUM
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node GPU_NUM                 
#SBATCH --cpus-per-task CPU_NUM
#SBATCH --gpu-bind=none


export NPROC_PER_NODE=GPU_NUM
export OMP_NUM_THREADS=8
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NCCL_DEBUG=INFO

echo ${MASTER_ADDR}:${MASTER_PORT}
echo $SLURM_NODEID

srun torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node $NPROC_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    source/train.py config/finetune.yaml
