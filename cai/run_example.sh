#!/bin/bash

# 示例运行脚本
# 请根据实际情况修改模型路径和数据路径

echo "开始生成 SFT 数据..."

# 单进程模式示例
echo "=== 单进程模式 ==="
python 1_gen_sft_data.py \
    --model_path "/path/to/your/model" \
    --data_path "../evaluation/Math-Benchmarks/data/math_500/test.jsonl" \
    --output_path "./output/sft_data_single.jsonl" \
    --critiques_path "prompts/critiques.json" \
    --few_shot_path "prompts/few_shot.txt" \
    --max_samples 10 \
    --temperature 0.7 \
    --max_tokens 2048 \
    --tensor_parallel_size 1 \
    --seed 42

echo "单进程模式完成！"

# 多进程模式示例
echo "=== 多进程模式 ==="
python 1_gen_sft_data.py \
    --model_path "/path/to/your/model" \
    --data_path "../evaluation/Math-Benchmarks/data/math_500/test.jsonl" \
    --output_path "./output/sft_data_multi.jsonl" \
    --critiques_path "prompts/critiques.json" \
    --few_shot_path "prompts/few_shot.txt" \
    --max_samples 20 \
    --temperature 0.7 \
    --max_tokens 2048 \
    --tensor_parallel_size 1 \
    --seed 42 \
    --use_multiprocessing \
    --num_processes 4 \
    --gpu_devices "0,1,2,3"

echo "多进程模式完成！"
echo "SFT 数据生成完成！"