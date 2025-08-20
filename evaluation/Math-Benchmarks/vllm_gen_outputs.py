from vllm import LLM, SamplingParams
import argparse
from data_loader import load_data
import random
import os
from datetime import datetime
from model_utils import load_hf_lm_and_tokenizer, generate_completions
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
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
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    # checkpoint
    parser.add_argument("--checkpoint_suffix", action="store_true", default=False)
    parser.add_argument("--prompt_file", default="", type=str)

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    if args.checkpoint_suffix:
        out_file_prefix += f"_cp{args.model_name_or_path.split('/')[-1]}"
    if args.prompt_file:
        out_file_prefix += f"_pf_{args.prompt_file.split('/')[-1].replace('.txt', '')}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
        )
        # tokenizer = None
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    # data_list.append("avg")
    # results.append(
    #     {
    #         "acc": sum([result["acc"] for result in results]) / len(results),
    #     }
    # )

    # print all results
    # pad = max([len(data_name) for data_name in data_list])
    # print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    # print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))

def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def main(llm, tokenizer, data_name, args):
    if args.prompt_file:
        PROMPT = read_txt(args.prompt_file)
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        question = example.get("question", None)
        if question is None:
            question = example.get("problem", None)
            example['question'] = question
            if question is not None:
                del example['problem']
        assert question is not None, "question is None, please check your data"

        if example["question"] == "":
            continue

        full_prompt = construct_prompt(example, data_name, args)

        # if data_name in ['aqua', 'mmlu_stem']:
        #     full_prompt += '\n' + 'This is a multiple-choice math problem. Please think step by step and finally select the final correct option from the given answer choices.\n'

        if idx == args.start:
            print(full_prompt)

        # gt_cot = example.get('solution', None)
        # if gt_cot is None:
        answer = example.get("answer", None)
        assert answer is not None, "answer is None, please check your data"

        sample = {
            "idx": idx,
            "question": example["question"],
            "answer": answer,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.prompt_file:
        input_prompts = [prompt+PROMPT for prompt in input_prompts]
    # else:
    #     input_prompts = [item[1] for item in input_prompts]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        # prompts = ['what is your name?', "how old are you", "Hello! What are you doing now?"]
        if args.use_vllm:
            # Avoid duplicate generation
            # Generate the first step first
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    # max_tokens=128,
                    n=1,
                    stop=stop_words, #+['## Step 2:'],
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )
            outputs = [output.outputs[0].text for output in outputs]
            # outputs = sorted(
            #     outputs, key=lambda x: int(x.request_id)
            # )  # sort outputs by request_id

            # step1s = [output.outputs[0].text for output in outputs]
            # Generate the remaining steps, and stop when the first step is generated again
            # outputs = llm.generate(
            #     [prompt + step1 for prompt, step1 in zip(prompts, step1s)],
            #     SamplingParams(
            #         temperature=args.temperature,
            #         top_p=args.top_p,
            #         max_tokens=args.max_tokens_per_call,
            #         n=1,
            #         stop=stop_words+['## Step 1:'],
            #         stop_token_ids=(
            #             [151645, 151643]
            #             if "qwen2" in args.model_name_or_path.lower()
            #             else None
            #         ),
            #     ),
            # )
            # outputs = sorted(
            #     outputs, key=lambda x: int(x.request_id)
            # )  # sort outputs by request_id
            # steps_left = [output.outputs[0].text for output in outputs]

            # if data_name in ["mmlu_stem", "aqua", "sat_math"]:
            #     outputs = llm.generate(
            #         [prompt + step1 + step_left+"\nSo let's choose the correct answer choice: " for prompt, step1, step_left in zip(prompts, step1s, steps_left)],
            #         SamplingParams(
            #             temperature=args.temperature,
            #             top_p=args.top_p,
            #             max_tokens=5,
            #             n=1,
            #             stop=stop_words,
            #             stop_token_ids=(
            #                 [151645, 151643]
            #                 if "qwen2" in args.model_name_or_path.lower()
            #                 else None
            #             ),
            #         ),
            #     )
            #     # outputs = sorted(
            #     #     outputs, key=lambda x: int(x.request_id)
            #     # )
            #     answer_choices = [output.outputs[0].text for output in outputs]
            #     outputs = [step1 + step_left +"So let's choose the correct answer choice: "+ answer_choice for step1, step_left, answer_choice in zip(step1s, steps_left, answer_choices)]
            # else:
            #     outputs = [step1 + step_left for step1, step_left in zip(step1s, steps_left)]

            
            # outputs = [output.outputs[0].text for output in outputs]
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        
        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            end_prompts.append((i, query))


    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    outputs_token_counter = []
    
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        # Logically, stop words themselves will not be generated
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)
        output_ids = tokenizer.encode(code, add_special_tokens=False)
        outputs_token_counter.append(len(output_ids))

    # extract preds

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        sample.pop("prompt")
        sample.update({"code": code})
        sample['code_len'] = len(code)
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
   
    for sample_i, sample in enumerate(all_samples):
        sample['token_counter'] = outputs_token_counter[sample_i]

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)

