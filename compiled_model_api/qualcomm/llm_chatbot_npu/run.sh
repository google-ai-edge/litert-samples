#!/bin/bash
unzip litertlm_sm8750.zip

ADB="adb -s RZCY21QJGCX"
export DEVICE_FOLDER=/data/local/tmp/gemma

$ADB push litertlm/* $DEVICE_FOLDER
$ADB shell chmod +x -R $DEVICE_FOLDER
$ADB shell LD_LIBRARY_PATH=$DEVICE_FOLDER ADSP_LIBRARY_PATH=$DEVICE_FOLDER \
    $DEVICE_FOLDER/litert_lm_main \
    --backend=npu \
    --model_path=$DEVICE_FOLDER/model.litertlm \
    --input_prompt \"Explain Qualcomm.\"
