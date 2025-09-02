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
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_DIR="/angel/fwk/code/SeRL/evaluation/Math-Benchmarks/data"
DATA_NAME="math_500"

N_SAMPLING=1
TEMPRATURE=0.6
MAX_TOKENS=32768
PROMPT_FILE=/angel/fwk/code/SeRL/evaluation/Math-Benchmarks/prompts/cot.txt
CHAT_TEMPLATE_ARG="--apply_chat_template"
# ================need to modify=======================

for MODEL_NAME_OR_PATH in "${MODEL_PATH_LIST[@]}"; do
    OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval_sampling_${N_SAMPLING}"
    
    python3 -u vllm_gen_outputs.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_name "${DATA_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --split "${SPLIT}" \
        --prompt_type "${PROMPT_TYPE}" \
        --num_test_sample "${NUM_TEST_SAMPLE}" \
        --seed 0 \
        --temperature ${TEMPRATURE} \
        --n_sampling ${N_SAMPLING} \
        --top_p 0.95 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --max_tokens_per_call ${MAX_TOKENS} \
        --overwrite \
        --data_dir "${DATA_DIR}" \
        --prompt_file ${PROMPT_FILE} \
        ${CHAT_TEMPLATE_ARG}
done

# End time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Total execution time: $execution_time seconds"
