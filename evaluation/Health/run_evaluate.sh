#!/bin/bash

# 定义基础路径和参数
BASE_PATH="/root/code/SeRL/evaluation/Health/outputs/root/code/models"
MODELS=(
    "/pubshare/fwk/code/SeRL/evaluation/Health/outputs/pubshare/fwk/orlhf_checkpoints/checkpoint/llama32_3B-reinforce_pp_med_rl/global_step100_hf"
    "/pubshare/fwk/code/SeRL/evaluation/Health/outputs/pubshare/fwk/orlhf_checkpoints/checkpoint/llama32_3B-reinforce_pp_med_rl/global_step200_hf"
    "/pubshare/fwk/code/SeRL/evaluation/Health/outputs/pubshare/fwk/orlhf_checkpoints/checkpoint/llama32_3B-reinforce_pp_med_rl/global_step300_hf"
    "/pubshare/fwk/code/SeRL/evaluation/Health/outputs/pubshare/fwk/orlhf_checkpoints/checkpoint/llama32_3B-reinforce_pp_med_rl/global_step400_hf"
)
DATASETS=("med_qa" "nephsap" "pubmedqa")
SUFFIX="_-1_seed0_t0.6_s0_e-1.jsonl"

# 动态生成文件列表
files=()
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        case $dataset in
            "med_qa")
                filename="medical_qa_medical_qa${SUFFIX}"
                ;;
            "nephsap")
                filename="nephsap_medical_qa${SUFFIX}"
                ;;
            "pubmedqa")
                filename="pubmedqa_pubmedqa${SUFFIX}"
                ;;
        esac
        files+=("${model}/${dataset}/${filename}")
    done
done

# 不再创建统一的报告目录，输出到原来的目录

# 遍历所有文件进行评估
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "正在评估: $file"
        
        # 提取文件名（不含路径和扩展名）
        basename=$(basename "$file" .jsonl)
        dirname=$(dirname "$file")
        model_name=$(basename $(dirname $(dirname "$file")))
        dataset_type=$(basename "$dirname")
        
        # 创建输出文件名（输出到原来的目录）
        report_prefix="${dirname}/${basename}"
        
        # 运行评估
        python evaluate.py "$file" \
            --output "${report_prefix}_report.json" \
            --summary "${report_prefix}_summary.json" \
            --enhanced-output "${report_prefix}_enhanced.jsonl" \
            --verbose
        
        echo "评估完成: $file"
        echo "---"
    else
        echo "文件不存在: $file"
    fi
done

echo "所有评估完成！"