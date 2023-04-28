#!/bin/bash

DATASET='MVTecAD/carpet'
SIZE=128
CHANNEL=3
GPU_ID=0

python3 src/main.py \
    --train2 \
    --dataset ${DATASET} \
    --size ${SIZE} \
    --nc ${CHANNEL} \
    --gpu_id ${GPU_ID} \
    --train2_iters 10000

