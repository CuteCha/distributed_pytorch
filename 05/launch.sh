#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 

#--------------------------------------------------
#SNMC
torchrun --nproc_per_node=2 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="172.18.0.11" \
        --master_port=1234 \
        main.py

#--------------------------------------------------
#MNMC
#master
torchrun --nproc_per_node=2 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr="172.18.0.11" \
        --master_port=1234 \
        main.py

#worker
torchrun --nproc_per_node=2 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr="172.18.0.11" \
        --master_port=1234 \
        main.py

#--------------------------------------------------