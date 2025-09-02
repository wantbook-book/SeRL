#!/bin/bash

# Start time
start_time=$(date +%s)

# ================need to modify=======================
OUTPUT_DIRS=(
    "/pubshare/fwk/code/SeRL/evaluation/Math-Benchmarks/outputs/pubshare/fwk/orlhf_checkpoints/select_checkpoint/qwen25_7B-random_bon_maj_bs16_seo/global_step100_hf/math_eval_sampling_16"
    "/pubshare/fwk/code/SeRL/evaluation/Math-Benchmarks/outputs/pubshare/fwk/orlhf_checkpoints/select_checkpoint/qwen25_7B-random_bon_maj_bs16_seo/global_step200_hf/math_eval_sampling_16"
    "/pubshare/fwk/code/SeRL/evaluation/Math-Benchmarks/outputs/pubshare/fwk/orlhf_checkpoints/select_checkpoint/qwen25_7B-random_bon_maj_bs16_seo_200/global_step50_hf/math_eval_sampling_16"
    "/pubshare/fwk/code/SeRL/evaluation/Math-Benchmarks/outputs/pubshare/fwk/orlhf_checkpoints/select_checkpoint/qwen25_7B-random_bon_maj_bs16_seo_200/global_step100_hf/math_eval_sampling_16"
)
SUBDIRS=("math_500" "math_hard" "asdiv" "college_math" "tabmwp")
FILE_NAME="test_pure_-1_seed0_t1.0_s0_e-1.jsonl"
# ================need to modify=======================

EVAL_SCRIPT="evaluate_outputs.py"
for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    # Check if the root directory exists
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Skipped $OUTPUT_DIR: Directory does not exist."
        continue
    fi
    # Iterate through subdirectories
    for SUBDIR in "${SUBDIRS[@]}"; do
        # Check if the subdirectory exists
        SUBDIR="$OUTPUT_DIR/$SUBDIR"
        if [ ! -d "$SUBDIR" ]; then
            echo "Skipped $SUBDIR: Directory does not exist."
            continue
        fi

        INPUT_JSONL="$SUBDIR/$FILE_NAME"
        if [ -f "$INPUT_JSONL" ]; then
            OUTPUT_JSONL="${INPUT_JSONL%.jsonl}_output.jsonl"
            echo "Evaluating: $INPUT_JSONL"
            python "$EVAL_SCRIPT" --input_jsonl "$INPUT_JSONL" --output_jsonl "$OUTPUT_JSONL" --gold_is_latex
        else
            echo "Skipped $SUBDIR: test.jsonl not found."
        fi
    done
done

# End time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"