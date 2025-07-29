from vllm import LLM, SamplingParams
import argparse
import random
import os
from datetime import datetime
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="./dataset/med_qa/test.jsonl", type=str, help="Path to the medical QA data file (supports MedQA and PubMedQA formats)")
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--prompt_type", default="medical_qa", type=str, help="Type of prompt to use (medical_qa or pubmedqa)")
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--prompt_file", default="", type=str)
    parser.add_argument("--include_options", action="store_true", help="Include answer options in the prompt")
    parser.add_argument("--dataset_type", default="medqa", type=str, choices=["medqa", "pubmedqa", "nephsap"], help="Type of dataset: medqa, pubmedqa, or nephsap")

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """Save data to JSONL file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def construct_medical_prompt(example, args):
    """Construct prompt for medical QA"""
    question = example["question"]
    
    if args.include_options and "options" in example:
        options = example["options"]
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        prompt = f"Question: {question}\n\nOptions:\n{options_text}\n\nPlease analyze this medical question step by step and put your final answer option (A, B, C, D, or E) within \\boxed{{}}"
    else:
        prompt = f"Question: {question}\n\nPlease analyze this medical question step by step and put your final answer within \\boxed{{}}."
    
    return prompt

def construct_pubmedqa_prompt(example, args):
    """Construct prompt for PubMedQA"""
    question = example["question"]
    context = example.get("context", "")
    
    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nBased on the provided context, please answer the question. Your answer should be one of: yes, no, or maybe. Please analyze the question step by step and put your final answer (yes, no, or maybe) within \\boxed{{}}."
    else:
        prompt = f"Question: {question}\n\nPlease answer the question. Your answer should be one of: yes, no, or maybe. Please analyze the question step by step and put your final answer (yes, no, or maybe) within \\boxed{{}}."
    
    return prompt

def extract_pubmedqa_answer(generated_text):
    """Extract yes/no/maybe answer from PubMedQA response"""
    import re
    
    # First try to extract from \boxed{}
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, generated_text, re.IGNORECASE)
    
    if boxed_matches:
        answer = boxed_matches[-1].strip().lower()
        if answer in ['yes', 'no', 'maybe']:
            return answer
    
    # If no boxed answer found, look for explicit yes/no/maybe in the text
    text_lower = generated_text.lower()
    
    # Look for patterns like "the answer is yes", "answer: no", etc.
    answer_patterns = [
        r'(?:the )?answer (?:is |: ?)(yes|no|maybe)',
        r'(?:final )?(?:answer|conclusion) (?:is |: ?)(yes|no|maybe)',
        r'\b(yes|no|maybe)\b(?:\.|$|\s*$)'
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            return matches[-1]
    
    # If still no clear answer, return the most likely based on keywords
    # if 'yes' in text_lower and 'no' not in text_lower:
    #     return 'yes'
    # elif 'no' in text_lower and 'yes' not in text_lower:
    #     return 'no'
    # elif 'maybe' in text_lower or 'uncertain' in text_lower or 'unclear' in text_lower:
    #     return 'maybe'
    
    return ''  # Default fallback

def prepare_data(args):
    """Prepare medical QA data"""
    # Load data
    examples = load_jsonl(args.data_file)
    
    # Add index if not present
    for i, example in enumerate(examples):
        if "idx" not in example:
            example["idx"] = i
    
    # Sample data if specified
    if args.num_test_sample > 0:
        examples = examples[:args.num_test_sample]
    
    # Shuffle if specified
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)
    
    # Select range
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]
    
    # Generate output file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    # Generate output file name based on dataset type or prompt type
    if args.dataset_type == "pubmedqa" or args.prompt_type == "pubmedqa":
        out_file_prefix = f"pubmedqa_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    elif args.dataset_type == "nephsap":
        out_file_prefix = f"nephsap_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    else:
        out_file_prefix = f"medical_qa_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    if args.prompt_file:
        out_file_prefix += f"_pf_{args.prompt_file.split('/')[-1].replace('.txt', '')}"
    
    # 构建包含模型路径的输出目录
    model_path_parts = args.model_name_or_path.strip('/').split('/')
    model_subdir = '/'.join(model_path_parts)
    output_dir = os.path.join(args.output_dir, model_subdir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine output subdirectory based on dataset type or prompt type
    if args.dataset_type == "pubmedqa" or args.prompt_type == "pubmedqa":
        output_subdir = "pubmedqa"
    elif args.dataset_type == "nephsap":
        output_subdir = "nephsap"
    else:
        output_subdir = "med_qa"
    out_file = f"{output_dir}/{output_subdir}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{output_subdir}", exist_ok=True)
    
    # Load processed samples if not overwriting
    processed_samples = []
    if not args.overwrite:
        if args.dataset_type == "pubmedqa" or args.prompt_type == "pubmedqa":
            output_subdir = "pubmedqa"
        elif args.dataset_type == "nephsap":
            output_subdir = "nephsap"
        else:
            output_subdir = "med_qa"
        if os.path.exists(f"{output_dir}/{output_subdir}/"):
            processed_files = [
                f for f in os.listdir(f"{output_dir}/{output_subdir}/")
                if f.endswith(".jsonl") and f.startswith(out_file_prefix)
            ]
            for f in processed_files:
                processed_samples.extend(load_jsonl(f"{output_dir}/{output_subdir}/{f}"))
    
    # Remove duplicates
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    
    return examples, processed_samples, out_file

def read_txt(file_path):
    """Read text file"""
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def setup_model(args):
    """Setup model and tokenizer"""
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
    else:
        # For non-vLLM usage, you would need to implement model loading here
        raise NotImplementedError("Non-vLLM model loading not implemented")
    
    return llm, tokenizer

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Setup model
    llm, tokenizer = setup_model(args)
    
    # Prepare data
    examples, processed_samples, out_file = prepare_data(args)
    
    print("=" * 50)
    if args.dataset_type == "pubmedqa" or args.prompt_type == "pubmedqa":
        dataset_name = "PubMedQA"
    elif args.dataset_type == "nephsap":
        dataset_name = "NephSAP"
    else:
        dataset_name = "Medical QA"
    print(f"{dataset_name} Data: {args.data_file}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Prompt Type: {args.prompt_type}")
    print(f"Remaining samples: {len(examples)}")
    if len(examples) > 0:
        print("Sample example:")
        print(examples[0])
    
    # Process examples
    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]
        question = example["question"]
        answer = example["answer"]
        
        # Construct prompt based on dataset type or prompt type
        if args.dataset_type == "pubmedqa" or args.prompt_type == "pubmedqa":
            full_prompt = construct_pubmedqa_prompt(example, args)
        else:
            # Both medqa and nephsap use the same medical_qa prompt format
            full_prompt = construct_medical_prompt(example, args)
        
        if args.prompt_file:
            PROMPT = read_txt(args.prompt_file)
            full_prompt = PROMPT + full_prompt
        
        if idx == args.start:
            print("Sample prompt:")
            print(full_prompt)
            print("=" * 50)
        
        sample = {
            "idx": idx,
            "question": question,
            "answer": answer,
            "prompt": full_prompt,
        }
        
        # Add additional fields
        for key in ["options", "meta_info", "answer_idx"]:
            if key in example:
                sample[key] = example[key]
        
        samples.append(sample)
    
    if len(samples) == 0:
        print("No new samples to process.")
        return
    
    # Prepare prompts for generation
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    
    # Setup stop words based on dataset type
    if args.prompt_type == "pubmedqa" or args.dataset_type == "pubmedqa":
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\nQuestion:", "\n\nContext:"]
    else:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\nQuestion:"]
    
    # Generate responses
    print("Starting generation...")
    start_time = time.time()
    
    if args.use_vllm:
        outputs = llm.generate(
            input_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_words,
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in args.model_name_or_path.lower()
                    else None
                ),
            ),
        )
        generated_texts = [output.outputs[0].text for output in outputs]
    else:
        raise NotImplementedError("Non-vLLM generation not implemented")
    
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Process outputs
    codes = []
    outputs_token_counter = []
    
    for i, generated_text in enumerate(generated_texts):
        # Clean up generated text
        code = generated_text.strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)
        
        # Count tokens
        output_ids = tokenizer.encode(code, add_special_tokens=False)
        outputs_token_counter.append(len(output_ids))
    
    # Combine results with samples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        sample.pop("prompt")
        
        # For PubMedQA, extract the final answer; medqa and nephsap use standard format
        if args.dataset_type == "pubmedqa" or args.prompt_type == "pubmedqa":
            extracted_answer = extract_pubmedqa_answer(code[0]) if code else 'maybe'
            sample.update({
                "generated_response": code,
                "extracted_answer": extracted_answer,
                "response_length": len(code[0]) if code else 0,
                "token_count": outputs_token_counter[i * args.n_sampling] if outputs_token_counter else 0
            })
        else:
            # Both medqa and nephsap use the same output format
            sample.update({
                "generated_response": code,
                "response_length": len(code[0]) if code else 0,
                "token_count": outputs_token_counter[i * args.n_sampling] if outputs_token_counter else 0
            })
        
        all_samples.append(sample)
    
    # Add processed samples
    all_samples.extend(processed_samples)
    
    # Save results
    if args.save_outputs:
        save_jsonl(all_samples, out_file)
        print(f"Results saved to: {out_file}")
    
    print(f"Processed {len(samples)} new samples")
    print(f"Total samples: {len(all_samples)}")

if __name__ == "__main__":
    main()