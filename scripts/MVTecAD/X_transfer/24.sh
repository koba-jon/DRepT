#!/bin/bash

IMAGE_PATH='datasets/MVTecAD/wood/train/000.png'
GMM_PATH='checkpoints/MVTecAD/carpet/stage2/models/carpet_testA_color_000.pth'
RESULT_PATH="transfer_result/2/4/"
SIZE=128
CHANNEL=3
GPU_ID=0

python3 src/main.py \
    --transfer \
    --transfer_image_path ${IMAGE_PATH} \
    --transfer_gmm_path ${GMM_PATH} \
    --transfer_result_dir ${RESULT_PATH} \
    --size ${SIZE} \
    --nc ${CHANNEL} \
    --gpu_id ${GPU_ID} \

