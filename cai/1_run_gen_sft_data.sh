#!/bin/bash

# 示例运行脚本
# 请根据实际情况修改模型路径和数据路径

echo "开始生成 SFT 数据..."

# python 1_gen_sft_data.py \
#     --model_path "/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct" \
#     --data_path "/pubshare/fwk/code/SeRL/openrlhf/dataset/math/train_with_idx.jsonl" \
#     --output_path "./output/sft_data.jsonl" \
#     --critiques_path "prompts/critiques.json" \
#     --few_shot_path "prompts/few_shot.txt" \
#     --max_samples 10 \
#     --temperature 0.7 \
#     --max_tokens 2048 \
#     --tensor_parallel_size 1 \
#     --seed 42

python 1_gen_sft_data.py \
    --model_path "/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct" \
    --data_path "/pubshare/fwk/code/SeRL/openrlhf/dataset/math/train_with_idx.jsonl" \
    --output_path "./output/sft_data_multi.jsonl" \
    --critiques_path "prompts/critiques.json" \
    --few_shot_path "prompts/few_shot.txt" \
    --max_samples 3000 \
    --temperature 0.7 \
    --max_tokens 2048 \
    --tensor_parallel_size 1 \
    --seed 42 \
    --use_multiprocessing \
    --num_processes 8 \
    --gpu_devices "0,1,2,3,4,5,6,7"

echo "SFT 数据生成完成！"