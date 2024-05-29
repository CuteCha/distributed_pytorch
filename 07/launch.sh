#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 
export MASTER_ADDR="172.18.0.11"
export MASTER_PORT="12355"
torchrun --nnodes=2 \
         --nproc_per_node=2 \
         --rdzv_id=100 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
         main.py \
         --bs 16 \
         --epochs=8 \
         --ckpt_path="tmp.pt" \
         --train_data_dir="./data"