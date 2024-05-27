#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 

#1 node, 4 GPUs per node (4GPUs)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    main.py