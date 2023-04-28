#!/bin/bash

SOURCE_DATASET=('MVTecAD/leather' 'MVTecAD/tile' 'MVTecAD/wood')
TARGET_DATASET='MVTecAD/carpet'

mkdir -p "checkpoints/${TARGET_DATASET}/stage2/models_as_source/"
for str in ${SOURCE_DATASET[@]}; do
    cp checkpoints/${str}/stage2/models/* checkpoints/${TARGET_DATASET}/stage2/models_for_target/
done

