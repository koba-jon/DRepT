#!/bin/bash

IMAGE_PATH='datasets/MVTecAD/leather/train/000.png'
GMM_PATH='checkpoints/MVTecAD/tile/stage2/models/tile_testA_oil_001.pth'
RESULT_PATH="transfer_result/1/2/"
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

