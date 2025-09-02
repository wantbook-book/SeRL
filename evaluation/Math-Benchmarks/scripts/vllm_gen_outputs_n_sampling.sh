#!/bin/bash
set -ex

# Start time
start_time=$(date +%s)

export TOKENIZERS_PARALLELISM=false
PROMPT_TYPE="pure"
NUM_TEST_SAMPLE=-1
SPLIT="test"

# ================need to modify=======================
# List of model paths
MODEL_PATH_LIST=(
    "/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-3b-0208seed_gen7500_online_filtered_0208_random_bon_maj_bs16/"
    "/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-3b-0208seed_gen7500_online_filtered_0208_random_bon_maj_bs16_iter1/"
    "/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-3b-0208seed_gen7500_online_filtered_0208_random_bon_maj_bs16_iter2/global_step100_hf"
    "/pubshare/fwk/orlhf_checkpoints/select_checkpoint/qwen25_7B-random_bon_maj_bs16_seo/global_step100_hf"
    "/pubshare/fwk/orlhf_checkpoints/select_checkpoint/qwen25_7B-random_bon_maj_bs16_seo/global_step200_hf"
    "/pubshare/fwk/orlhf_checkpoints/select_checkpoint/qwen25_7B-random_bon_maj_bs16_seo_200/global_step50_hf"
    "/pubshare/fwk/orlhf_checkpoints/select_checkpoint/qwen25_7B-random_bon_maj_bs16_seo_200/global_step100_hf"
)
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_SAMPLING=16
DATA_DIR="/pubshare/fwk/code/SeRL/evaluation/Math-Benchmarks/data"
DATA_NAME="math_500,math_hard,asdiv,college_math,tabmwp"
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
        --temperature 1 \
        --n_sampling ${N_SAMPLING} \
        --top_p 0.95 \
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
