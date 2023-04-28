#!/bin/bash

SOURCE_DATASET=('MVTecAD/carpet' 'MVTecAD/tile' 'MVTecAD/wood')
TARGET_DATASET='MVTecAD/leather'

mkdir -p "checkpoints/${TARGET_DATASET}/stage2/models_for_target/"
for str in ${SOURCE_DATASET[@]}; do
    cp checkpoints/${str}/stage2/models/* checkpoints/${TARGET_DATASET}/stage2/models_for_target/
done

