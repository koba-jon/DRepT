#!/bin/bash

DATASET='MVTecAD/leather'
SIZE=128
CHANNEL=3
GPU_ID=0
GLOBAL_LOOP=1
LOCAL_LOOP=100

python3 src/main.py \
    --train3 \
    --dataset ${DATASET} \
    --size ${SIZE} \
    --nc ${CHANNEL} \
    --gpu_id ${GPU_ID} \
    --train3_global_loops ${GLOBAL_LOOP} \
    --train3_local_loops ${LOCAL_LOOP} \
    --train3_batch_size 16

