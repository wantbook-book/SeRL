import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
import debugpy
# debugpy.listen(('localhost', 5678))
# debugpy.wait_for_client()

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 40000
random.seed(12345)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data

def load_gpqa():
    dataset = load_jsonl('./dataset/gpqa_diamond.jsonl')
    dataset = preprocess(dataset)
    return dataset


def load_model():
    llm = LLM(model=args.model, gpu_memory_utilization=float(args.gpu_util),
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=max_model_length,
                trust_remote_code=True)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, top_p=args.top_p,
                                        stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return (llm, sampling_params), tokenizer


def preprocess(dataset):
    res_dataset = []
    for each in dataset:
        options = each['options']
        random.shuffle(options)
        each["options"] = options
        each["answer_index"] = choices[options.index(each["answer"])]
        res_dataset.append(each)
    return res_dataset


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(dataset, curr, k):
    prompt = ""
    with open(f"prompts/cot.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    if k > 0:
        val_df = dataset[: k]
        for example in val_df:
            prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        for each in res:
            fo.write(json.dumps(each) + "\n")
    for each in res:
        if not each["pred"]:
            # x = random.randint(0, len(each["options"]) - 1)
            # if x == each["answer_index"]:
            #     corr += 1
            #     # print("random hit.")
            # else:
            wrong += 1
        elif each["pred"] == each["answer_index"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(model, tokenizer, dataset, output_path):
    llm, sampling_params = model
    global choices
    inference_batches = []

    for i in tqdm(range(len(dataset))):
        k = args.ntrain
        curr = dataset[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(dataset, curr, k)
            if args.apply_chat_template:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - args.max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    res = []
    for j, curr in enumerate(dataset):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


def main():
    model, tokenizer = load_model()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    dataset = load_gpqa()
    sta_dict = {}
    
    output_path = os.path.join(save_result_dir, f"outputs.jsonl")
    acc, corr_count, wrong_count = eval_cot(model, tokenizer, dataset, output_path)
    summary_path = os.path.join(save_result_dir, f"metric.json")
    with open(summary_path, 'w') as f:
        json.dump({"accu": acc, "corr": corr_count, "wrong": wrong_count}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--max_new_tokens", default=1024, type=int)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_result_dir = os.path.join(
        args.save_dir, args.model[1:] if args.model[0] == '/' else args.model
    )
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    os.makedirs(save_result_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    main()


