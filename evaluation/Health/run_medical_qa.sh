#!/bin/bash

# 医学问答数据集生成脚本示例
# 基于 vLLM 生成医学问答回答

# 设置基本参数
MODEL_PATH="/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct"  # 替换为实际的模型路径
DATA_FILE="./dataset/med_qa/test.jsonl"
OUTPUT_DIR="./outputs"
temperature=0.6
top_p=0.95


# 基本运行示例
# echo "运行医学问答生成 - 基本模式"
# python vllm_generate.py \
#     --data_file $DATA_FILE \
#     --model_name_or_path $MODEL_PATH \
#     --output_dir $OUTPUT_DIR \
#     --use_vllm \
#     --save_outputs \
#     --temperature $temperature \
#     --top_p $top_p \
#     --max_tokens_per_call 1024 \
#     --num_test_sample 10

# 包含选项的运行示例
echo "运行医学问答生成 - 包含选项模式"
python vllm_generate.py \
    --data_file $DATA_FILE \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --use_vllm \
    --save_outputs \
    --include_options \
    --temperature $temperature \
    --top_p $top_p \
    --max_tokens_per_call 1024 \
    --num_test_sample -1

# 使用聊天模板的运行示例
# echo "运行医学问答生成 - 聊天模板模式"
# python vllm_generate.py \
#     --data_file $DATA_FILE \
#     --model_name_or_path $MODEL_PATH \
#     --output_dir $OUTPUT_DIR \
#     --use_vllm \
#     --save_outputs \
#     --apply_chat_template \
#     --include_options \
#     --temperature $temperature \
#     --top_p $top_p \
#     --max_tokens_per_call 1024 \
#     --num_test_sample 10

# 完整数据集运行示例
# echo "运行医学问答生成 - 完整数据集"
# python vllm_generate.py \
#     --data_file $DATA_FILE \
#     --model_name_or_path $MODEL_PATH \
#     --output_dir $OUTPUT_DIR \
#     --use_vllm \
#     --save_outputs \
#     --include_options \
#     --apply_chat_template \
#     --temperature $temperature \
#     --top_p $top_p \
#     --max_tokens_per_call 1024 \
#     --num_test_sample -1

echo "医学问答生成完成！"