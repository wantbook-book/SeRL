#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=0
# 定义模型列表
models=(
    "/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct"
    "/pubshare/LLM/Qwen/Qwen2.5-7B-Instruct"
    "/pubshare/fwk/orlhf_checkpoints/checkpoint/llama32_3B-reinforce_pp_med_rl/global_step100_hf"
    "/pubshare/fwk/orlhf_checkpoints/checkpoint/llama32_3B-reinforce_pp_med_rl/global_step200_hf"
    "/pubshare/fwk/orlhf_checkpoints/checkpoint/llama32_3B-reinforce_pp_med_rl/global_step300_hf"
    "/pubshare/fwk/orlhf_checkpoints/checkpoint/llama32_3B-reinforce_pp_med_rl/global_step400_hf"
)

# 定义数据集配置
datasets=(
    "medqa:./dataset/med_qa/test.jsonl:medical_qa"
    "pubmedqa:./dataset/pubmedqa/test.jsonl:pubmedqa"
    "nephsap:./dataset/nephsap/test.jsonl:medical_qa"
)

# 遍历所有模型和数据集组合
for model in "${models[@]}"; do
    for dataset_config in "${datasets[@]}"; do
        IFS=':' read -r dataset_type data_file prompt_type <<< "$dataset_config"
        
        echo "Running $model on $dataset_type..."
        python vllm_generate.py \
            --data_file "$data_file" \
            --model_name_or_path "$model" \
            --dataset_type "$dataset_type" \
            --prompt_type "$prompt_type" \
            --use_vllm \
            --save_outputs \
            --temperature 0.6 \
            --top_p 0.95 \
            --max_tokens_per_call 2048 \
            --output_dir ./outputs \
            --overwrite \
            --include_options \
            
    done
done