#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
dirI_top="${SCRIPT_DIR}/mvtec_anomaly_detection"
dirO_top="${SCRIPT_DIR}/MVTecAD"
dirO_train="train"
dirO_train_anomaly="trainA"
dirO_train_normal="trainN"
dirO_test_anomaly="testA"
dirO_test_anomaly_GT="testGT"
dirO_test_normal="testN"

dirs_class=`find ${dirI_top}/* -maxdepth 0 -type d`
mkdir ${dirO_top}


# Make directories
for dir in ${dirs_class};
do
    dir_class=`basename ${dir}`
    mkdir "${dirO_top}/${dir_class}"
done


# Link training files
for dirI in ${dirs_class};
do

    dir_class=`basename ${dirI}`

    # Make directories
    train="${dirO_top}/${dir_class}/${dirO_train}"
    trainA="${dirO_top}/${dir_class}/${dirO_train_anomaly}"
    trainN="${dirO_top}/${dir_class}/${dirO_train_normal}"
    testA="${dirO_top}/${dir_class}/${dirO_test_anomaly}"
    testGT="${dirO_top}/${dir_class}/${dirO_test_anomaly_GT}"
    testN="${dirO_top}/${dir_class}/${dirO_test_normal}"
    mkdir ${train}
    mkdir ${trainA}
    mkdir ${trainN}
    mkdir ${testA}
    mkdir ${testGT}
    mkdir ${testN}

    # training data
    count=0
    files_train=`find ${dirI}/train/*/* -maxdepth 0 -type f`
    for f in ${files_train};
    do
        num=$(printf "%03d" "${count}")
        ln -s ${f} "${train}/${num}.png"
        ln -s ${f} "${trainN}/${dir_class}_${dirO_train}_${num}.png"
        count=`expr ${count} + 1`
    done


    # test data (anomaly)
    dirs_anomaly=`find ${dirI}/test/* -maxdepth 0 -type d -not -path "${dirI}/test/good"`
    for dir_anomaly in ${dirs_anomaly};
    do
        count=0
        anomaly=`basename ${dir_anomaly}`
        files_anomaly=`find ${dirI}/test/${anomaly}/* -maxdepth 0 -type f`
        for f in ${files_anomaly};
        do
            num=$(printf "%03d" "${count}")
            ln -s ${f} "${testA}/${anomaly}_${num}.png"
            ln -s ${f} "${trainA}/${dir_class}_${dirO_test_anomaly}_${anomaly}_${num}.png"
            count=`expr ${count} + 1`
        done
    done


    # test data (anomaly-GT)
    dirs_anomaly_GT=`find ${dirI}/ground_truth/* -maxdepth 0 -type d`
    for dir_anomaly_GT in ${dirs_anomaly_GT};
    do
        count=0
        anomaly_GT=`basename ${dir_anomaly_GT}`
        files_anomaly_GT=`find ${dirI}/ground_truth/${anomaly_GT}/* -maxdepth 0 -type f`
        for f in ${files_anomaly_GT};
        do
            num=$(printf "%03d" "${count}")
            ln -s ${f} "${testGT}/${anomaly_GT}_${num}.png"
            count=`expr ${count} + 1`
        done
    done


    # test data (normal)
    count=0
    files_normal=`find ${dirI}/test/good/* -maxdepth 0 -type f`
    for f in ${files_normal};
    do
        num=$(printf "%03d" "${count}")
        ln -s ${f} "${testN}/${num}.png"
        ln -s ${f} "${trainN}/${dir_class}_${dirO_test_normal}_${num}.png"
        count=`expr ${count} + 1`
    done


done
