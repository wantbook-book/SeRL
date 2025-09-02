#!/bin/bash

# Start time
start_time=$(date +%s)

# ================CAI输出文件评估脚本=======================
# 专门用于评估CAI生成的initial和revision结果文件

# 定义要评估的文件
INPUT_FILES=(
    "/pubshare/fwk/code/SeRL/cai/output/llama32_3b_math500_initial_results.jsonl"
    "/pubshare/fwk/code/SeRL/cai/output/llama32_3b_math500_revision_results.jsonl"
)

# 评估脚本路径
EVAL_SCRIPT="evaluate_outputs.py"

# 检查评估脚本是否存在
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "错误: 评估脚本 $EVAL_SCRIPT 不存在"
    echo "请确保在正确的目录下运行此脚本"
    exit 1
fi

echo "开始评估CAI输出文件..."
echo "==========================================="

# 遍历每个输入文件
for INPUT_JSONL in "${INPUT_FILES[@]}"; do
    # 检查输入文件是否存在
    if [ ! -f "$INPUT_JSONL" ]; then
        echo "跳过 $INPUT_JSONL: 文件不存在"
        continue
    fi
    
    # 生成输出文件名
    OUTPUT_JSONL="${INPUT_JSONL%.jsonl}_output.jsonl"
    METRICS_FILE="${INPUT_JSONL%.jsonl}_metrics.json"
    
    echo "正在评估: $INPUT_JSONL"
    echo "输出文件: $OUTPUT_JSONL"
    echo "指标文件: $METRICS_FILE"
    
    # 运行评估
    python "$EVAL_SCRIPT" --input_jsonl "$INPUT_JSONL" --output_jsonl "$OUTPUT_JSONL" --gold_is_latex
    
    # 检查评估是否成功
    if [ $? -eq 0 ]; then
        echo "✓ 评估完成: $INPUT_JSONL"
        
        # 如果生成了指标文件，显示准确率
        if [ -f "$METRICS_FILE" ]; then
            echo "指标文件已生成: $METRICS_FILE"
            # 尝试提取准确率信息
            if command -v jq &> /dev/null; then
                accuracy=$(jq -r '.accuracy // "N/A"' "$METRICS_FILE" 2>/dev/null)
                echo "准确率: $accuracy"
            fi
        fi
    else
        echo "✗ 评估失败: $INPUT_JSONL"
    fi
    
    echo "------------------------------------------"
done

# End time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "==========================================="
echo "评估完成!"
echo "总执行时间: $execution_time 秒"

# 显示生成的文件
echo ""
echo "生成的文件:"
for INPUT_JSONL in "${INPUT_FILES[@]}"; do
    if [ -f "$INPUT_JSONL" ]; then
        OUTPUT_JSONL="${INPUT_JSONL%.jsonl}_output.jsonl"
        METRICS_FILE="${INPUT_JSONL%.jsonl}_metrics.json"
        
        if [ -f "$OUTPUT_JSONL" ]; then
            echo "  - $OUTPUT_JSONL"
        fi
        if [ -f "$METRICS_FILE" ]; then
            echo "  - $METRICS_FILE"
        fi
    fi
done