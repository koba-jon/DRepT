#!/bin/bash

SOURCE_DATASET=('MVTecAD/carpet' 'MVTecAD/leather' 'MVTecAD/tile')
TARGET_DATASET='MVTecAD/wood'

mkdir -p "checkpoints/${TARGET_DATASET}/stage2/models_as_source/"
for str in ${SOURCE_DATASET[@]}; do
    cp checkpoints/${str}/stage2/models/* checkpoints/${TARGET_DATASET}/stage2/models_as_source/
done

