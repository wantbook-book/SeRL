#!/bin/bash

# 定义要评估的文件列表
files=(
    # "/pubshare/fwk/code/SeRL/evaluation/Health/outputs/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct/nephsap/nephsap_medical_qa_-1_seed0_t0.0_s0_e-1.jsonl"
    # "/pubshare/fwk/code/SeRL/evaluation/Health/outputs/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct/pubmedqa/pubmedqa_pubmedqa_-1_seed0_t0.0_s0_e-1.jsonl"
    # "/pubshare/fwk/code/SeRL/evaluation/Health/outputs/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct/med_qa/medical_qa_medical_qa_-1_seed0_t0.0_s0_e-1.jsonl"
    "/pubshare/fwk/code/SeRL/evaluation/Health/outputs/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct/med_qa/medical_qa_medical_qa_-1_seed0_t0.6_s0_e-1.jsonl"
)

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