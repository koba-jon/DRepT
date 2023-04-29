#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

bash ${SCRIPT_DIR}/A_stage1/carpet.sh
bash ${SCRIPT_DIR}/A_stage1/leather.sh
bash ${SCRIPT_DIR}/A_stage1/tile.sh
bash ${SCRIPT_DIR}/A_stage1/wood.sh

bash ${SCRIPT_DIR}/B_stage2/carpet.sh
bash ${SCRIPT_DIR}/B_stage2/leather.sh
bash ${SCRIPT_DIR}/B_stage2/tile.sh
bash ${SCRIPT_DIR}/B_stage2/wood.sh

bash ${SCRIPT_DIR}/C_set/carpet.sh
bash ${SCRIPT_DIR}/C_set/leather.sh
bash ${SCRIPT_DIR}/C_set/tile.sh
bash ${SCRIPT_DIR}/C_set/wood.sh

bash ${SCRIPT_DIR}/D_stage3/carpet.sh
bash ${SCRIPT_DIR}/D_stage3/leather.sh
bash ${SCRIPT_DIR}/D_stage3/tile.sh
bash ${SCRIPT_DIR}/D_stage3/wood.sh

bash ${SCRIPT_DIR}/E_test/carpet.sh
bash ${SCRIPT_DIR}/E_test/leather.sh
bash ${SCRIPT_DIR}/E_test/tile.sh
bash ${SCRIPT_DIR}/E_test/wood.sh

