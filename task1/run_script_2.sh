#!/bin/bash

set -e


MODEL_CHOICE="google/mt5-xl"
LORA_TARGET_MODULES="q v"

BASE_OUTPUT_DIR="./training_results"
NUM_TRAIN_EPOCHS=5
MAX_SEQ_LENGTH=1024
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=5
GRADIENT_ACCUMULATION_STEPS=2
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
WARMUP_RATIO=0.1
LORA_R=64
LORA_ALPHA=128
SEED=42


ALL_TRAIN_DATASETS=(ces.rst.crdt deu.rst.pcc eng.dep.scidtb eng.erst.gum eng.rst.rstdt eng.sdrt.stac eng.rst.umuc eng.rst.oll eng.rst.sts eng.sdrt.msdc eus.rst.ert fas.rst.prstc fra.sdrt.annodis fra.sdrt.summre nld.rst.nldt por.rst.cstn rus.rst.rrt spa.rst.rststb spa.rst.sctb zho.dep.scidtb zho.rst.gcdt zho.rst.sctb)
ALL_TEST_DATASETS=(eng.dep.scidtb rus.rst.rrt eng.etst.gentle)

ZHO_TRAIN_DATASETS=()
ALPHABET_TRAIN_DATASETS=()
ZHO_TEST_DATASETS=()
ALPHABET_TEST_DATASETS=()

for dataset in "${ALL_TRAIN_DATASETS[@]}"; do
    if [[ $dataset == zho.* ]]; then
        ZHO_TRAIN_DATASETS+=("$dataset")
    else
        ALPHABET_TRAIN_DATASETS+=("$dataset")
    fi
done

for dataset in "${ALL_TEST_DATASETS[@]}"; do
    if [[ $dataset == zho.* ]]; then
        ZHO_TEST_DATASETS+=("$dataset")
    else
        ALPHABET_TEST_DATASETS+=("$dataset")
    fi
done


run_experiment() {
    local EXP_NAME=$1
    local TRAIN_DS=$2
    local TEST_DS=$3
    local USE_WEIGHTED_LOSS=$4
    local USE_FGM=$5
    local FGM_EPSILON=$6
    local LEARNING_RATE=$7
    local DROPOUT=$8

    echo "========================================================================"
    echo ">> Running Experiment: ${EXP_NAME}"
    echo ">> Training on: ${TRAIN_DS}"
    echo ">> Testing on: ${TEST_DS}"
    echo "========================================================================"

    local output_suffix="${MODEL_CHOICE##*/}_${EXP_NAME}"
    local CURRENT_OUTPUT_MODEL_PATH="${BASE_OUTPUT_DIR}/${output_suffix}/"
    
    mkdir -p "${CURRENT_OUTPUT_MODEL_PATH}"

    local cmd=(
        python3 seg_batch.py
        --do_train
        --do_test
        --train_datasets ${TRAIN_DS}
        --test_datasets ${TEST_DS}
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
}


non_zho_train_str="${ALPHABET_TRAIN_DATASETS[*]}"
non_zho_test_str="${ALPHABET_TEST_DATASETS[*]}"



run_experiment \
    "script2_results" \
    "${non_zho_train_str}" \
    "${non_zho_test_str}" \
    false \
    false \
    0.5 \
    1e-4 \
    0.1



echo "All experiments completed successfully."