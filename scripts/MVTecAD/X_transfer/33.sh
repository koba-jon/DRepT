IMAGE_PATH='datasets/MVTecAD/tile/train/000.png'
GMM_PATH='checkpoints/MVTecAD/leather/stage2/models/leather_testA_glue_015.pth'
RESULT_PATH="transfer_result/3/3/"
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

