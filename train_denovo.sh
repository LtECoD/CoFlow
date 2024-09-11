#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate esm3

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=8

torchrun --nproc_per_node $NPROC_PER_NODE \
    source/train.py config/denovo.yaml