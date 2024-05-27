#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 
torchrun --nproc_per_node=2 main.py