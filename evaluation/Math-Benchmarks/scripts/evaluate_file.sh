EVAL_SCRIPT="evaluate_outputs.py"
INPUT_JSONL=/angel/fwk/code/SAE-Reasoning/evaluation/deepseek-llama-8b-math-500/deepseek-ai__DeepSeek-R1-Distill-Llama-8B/samples_math-500_2025-08-13T08-53-03.659236_eval2.jsonl
OUTPUT_JSONL="${INPUT_JSONL%.jsonl}_output.jsonl"
python "$EVAL_SCRIPT" --input_jsonl "$INPUT_JSONL" --output_jsonl "$OUTPUT_JSONL" --gold_is_latex