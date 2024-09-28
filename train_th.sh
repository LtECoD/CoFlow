#! /bin/bash

#SBATCH --job-name tune
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 4                   
#SBATCH --cpus-per-task 4
#SBATCH --gpu-bind=none
#SBATCH --output ./log/coflow_continue.out
#SBATCH --error ./log/coflow_continue.err
#SBATCH --exclusive

module load cuda/12.4
eval "$(conda shell.bash hook)"
conda activate esm3

export NPROC_PER_NODE=4
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
    source/train.py config/finetune_continue.yaml

