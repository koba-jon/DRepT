#!/bin/bash

DATASET='MVTecAD/tile'
SIZE=128
CHANNEL=3
GPU_ID=0
ITERATION=10000

python3 src/main.py \
    --train2 \
    --dataset ${DATASET} \
    --size ${SIZE} \
    --nc ${CHANNEL} \
    --gpu_id ${GPU_ID} \
    --train2_iters ${ITERATION}

