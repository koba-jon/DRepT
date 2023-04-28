#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

bash ${SCRIPT_DIR}/1_stage1/carpet.sh
bash ${SCRIPT_DIR}/1_stage1/leather.sh
bash ${SCRIPT_DIR}/1_stage1/tile.sh
bash ${SCRIPT_DIR}/1_stage1/wood.sh

bash ${SCRIPT_DIR}/2_stage2/carpet.sh
bash ${SCRIPT_DIR}/2_stage2/leather.sh
bash ${SCRIPT_DIR}/2_stage2/tile.sh
bash ${SCRIPT_DIR}/2_stage2/wood.sh

bash ${SCRIPT_DIR}/3_set/carpet.sh
bash ${SCRIPT_DIR}/3_set/leather.sh
bash ${SCRIPT_DIR}/3_set/tile.sh
bash ${SCRIPT_DIR}/3_set/wood.sh

bash ${SCRIPT_DIR}/4_stage3/carpet.sh
bash ${SCRIPT_DIR}/4_stage3/leather.sh
bash ${SCRIPT_DIR}/4_stage3/tile.sh
bash ${SCRIPT_DIR}/4_stage3/wood.sh

bash ${SCRIPT_DIR}/5_test/carpet.sh
bash ${SCRIPT_DIR}/5_test/leather.sh
bash ${SCRIPT_DIR}/5_test/tile.sh
bash ${SCRIPT_DIR}/5_test/wood.sh

