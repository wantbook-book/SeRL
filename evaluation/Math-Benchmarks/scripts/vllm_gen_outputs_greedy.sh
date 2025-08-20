#!/bin/bash
set -ex

# Start time
start_time=$(date +%s)
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
PROMPT_TYPE="pure"
NUM_TEST_SAMPLE=-1
SPLIT="test"

# ================need to modify=======================
# List of model paths
MODEL_PATH_LIST=(
    "/angel/fwk/checkpoints/qwen25_7B-serl_iter5"
    "/angel/fwk/checkpoints/qwen25_7B-serl_iter5/global_step50_hf"
)
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_DIR="/root/code/SeRL/evaluation/Math-Benchmarks/data"
DATA_NAME="math_500,math_hard,asdiv,college_math,tabmwp"
# ================need to modify=======================
N_SAMPLING=1

for MODEL_NAME_OR_PATH in "${MODEL_PATH_LIST[@]}"; do
    OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval_greedy"
    
    python3 -u vllm_gen_outputs.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_name "${DATA_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --split "${SPLIT}" \
        --prompt_type "${PROMPT_TYPE}" \
        --num_test_sample "${NUM_TEST_SAMPLE}" \
        --seed 0 \
        --temperature 0 \
        --n_sampling ${N_SAMPLING} \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --overwrite \
        --data_dir "${DATA_DIR}"
done

# End time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Total execution time: $execution_time seconds"
