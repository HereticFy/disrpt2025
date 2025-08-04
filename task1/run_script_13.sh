#!/bin/bash

set -e


MODEL_CHOICE="google/mt5-xl"
LORA_TARGET_MODULES="q v"

TRAIN_DATASETS="ces.rst.crdt deu.rst.pcc eng.dep.scidtb eng.erst.gum eng.rst.rstdt eng.sdrt.stac eng.rst.umuc eng.rst.oll eng.rst.sts eng.sdrt.msdc eus.rst.ert fas.rst.prstc fra.sdrt.annodis fra.sdrt.summre nld.rst.nldt por.rst.cstn rus.rst.rrt spa.rst.rststb spa.rst.sctb zho.dep.scidtb zho.rst.gcdt zho.rst.sctb"
TEST_DATASETS="eng.rst.sts"

BASE_OUTPUT_DIR="./training_results"
NUM_TRAIN_EPOCHS=5
MAX_SEQ_LENGTH=1024
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
DEFAULT_LEARNING_RATE=3e-5
DEFAULT_DROPOUT=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
WARMUP_RATIO=0.1
LORA_R=64
LORA_ALPHA=128
SEED=42

run_experiment() {
    local EXP_NAME=$1
    local USE_WEIGHTED_LOSS=$2
    local USE_FGM=$3
    local FGM_EPSILON=$4
    local LEARNING_RATE=$5  
    local DROPOUT=$6

    echo "========================================================================"
    echo "Running Experiment: ${EXP_NAME}"
    echo "Model: ${MODEL_CHOICE}, LR: ${LEARNING_RATE}, Dropout: ${DROPOUT}"
    echo "Weighted Loss: ${USE_WEIGHTED_LOSS}, FGM: ${USE_FGM}, Epsilon: ${FGM_EPSILON}"
    echo "========================================================================"

    local train_langs_str=$(echo ${TRAIN_DATASETS} | tr ' ' '+')
    local output_suffix="${MODEL_CHOICE##*/}_${EXP_NAME}"
    local CURRENT_OUTPUT_MODEL_PATH="${BASE_OUTPUT_DIR}/${output_suffix}/"
    
    mkdir -p "${CURRENT_OUTPUT_MODEL_PATH}"

    local cmd=(
        python3 seg_batch.py
        --do_train
        --do_test
        --train_datasets ${TRAIN_DATASETS}
        --test_datasets ${TEST_DATASETS}
        --model_choice="${MODEL_CHOICE}"
        --output_model_path="${CURRENT_OUTPUT_MODEL_PATH}"
        --lora_target_modules ${LORA_TARGET_MODULES}
        --num_train_epochs="${NUM_TRAIN_EPOCHS}"
        --max_seq_length="${MAX_SEQ_LENGTH}"
        --train_batch_size="${TRAIN_BATCH_SIZE}"
        --eval_batch_size="${EVAL_BATCH_SIZE}"
        --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
        --learning_rate="${LEARNING_RATE}"
        --dropout="${DROPOUT}"
        --weight_decay="${WEIGHT_DECAY}"
        --max_grad_norm="${MAX_GRAD_NORM}"
        --warmup_ratio="${WARMUP_RATIO}"
        --lora_r="${LORA_R}"
        --lora_alpha="${LORA_ALPHA}"
        --seed="${SEED}"
    )

    if [ "$USE_WEIGHTED_LOSS" = true ]; then
        cmd+=(--use_weighted_loss)
    fi
    if [ "$USE_FGM" = true ]; then
        cmd+=(--use_fgm --fgm_epsilon="${FGM_EPSILON}")
    fi

    echo "Executing command..."
    "${cmd[@]}"

    if [ $? -ne 0 ]; then
        echo "Error: Command failed for experiment ${EXP_NAME}. Exiting."
        exit 1
    fi

    echo "Finished experiment: ${EXP_NAME}"
    echo ""
}

echo "Starting all experiments..."

run_experiment "script13_results" false false 0.0 5e-5 0.1


echo "All experiments completed successfully."