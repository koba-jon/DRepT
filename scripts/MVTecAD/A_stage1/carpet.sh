#!/bin/bash

DATASET='MVTecAD/carpet'
SIZE=128
CHANNEL=3
GPU_ID=0
GLOBAL_LOOP=100
LOCAL_LOOP=100

python3 src/main.py \
    --train1 \
    --dataset ${DATASET} \
    --size ${SIZE} \
    --nc ${CHANNEL} \
    --gpu_id ${GPU_ID} \
    --train1_global_loops ${GLOBAL_LOOP} \
    --train1_local_loops ${LOCAL_LOOP} \
    --train1_batch_size 16

