#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 

torchrun --nproc_per_node=2 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="127.0.0.1" \
        --master_port=1234 \
        main.py