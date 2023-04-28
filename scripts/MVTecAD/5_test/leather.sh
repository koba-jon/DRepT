#!/bin/bash

DATASET='MVTecAD/leather'
SIZE=128
CHANNEL=3
GPU_ID=0

python3 src/main.py \
    --test \
    --dataset ${DATASET} \
    --size ${SIZE} \
    --nc ${CHANNEL} \
    --gpu_id ${GPU_ID} \
    --test_result_dir "test_result/${DATASET}"
