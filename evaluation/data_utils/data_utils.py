import re
import json
from collections import defaultdict, Counter
import random
from math_verify import LatexExtractionConfig, ExprExtractionConfig, math_metric
from pathlib import Path
def create_dpo_dataset(input_file, reward_key, output_file):
    data_list = []
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            rewards = entry[reward_key]
            # Find the maximum reward and the minimum reward to form a pair of data.
            min_reward_index = rewards.index(min(rewards))
            max_reward_index = rewards.index(max(rewards))
            question = entry.get("problem", "")
            if question == "":
                question = entry.get("question", "")
            assert question != "", "question is empty"
            data_item = {
                "idx": entry["idx"],
                "instruction": question,
                "input": "",
                "answer": entry["answer"],
                "min_reward": rewards[min_reward_index],
                "max_reward": rewards[max_reward_index],
                "chosen": entry['code'][max_reward_index],
                'rejected': entry['code'][min_reward_index],
            }

            if reward_key == 'rule_rewards' and rewards[max_reward_index] != 1:
                gt_cot = entry.get("solution", "")
                assert gt_cot != "", "gt_cot is empty"
                data_item["chosen"] = gt_cot
                data_item["max_reward"] = 100
                
            data_list.append(data_item)
            
    # Save the preprocessed data to a new JSON file
    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + '\n')
    exit()

def create_sft_dataset(input_file, reward_key, output_file):
    data_list = []
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            rewards = entry[reward_key]
            # Find the maximum reward and the minimum reward to form a pair of data.
            min_reward_index = rewards.index(min(rewards))
            max_reward_index = rewards.index(max(rewards))
            question = entry.get("problem", "")
            if question == "":
                question = entry.get("question", "")
            assert question != "", "question is empty"
            data_item = {
                "idx": entry["idx"],
                "instruction": question,
                "input": "",
                "output": entry['code'][max_reward_index],
            }

            if reward_key == 'rule_rewards' and rewards[max_reward_index] != 1:
                gt_cot = entry.get("solution", "")
                assert gt_cot != "", "gt_cot is empty"
                data_item["output"] = gt_cot
                
            data_list.append(data_item)
            
    # Save the preprocessed data to a new JSON file
    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + '\n')
    exit()

def group_pred(preds):
    cnt = Counter(preds)
    majority = cnt.most_common(1)[0][0]
    groups = defaultdict(list)
    for idx, pred in enumerate(preds):
        groups[pred].append(idx)
    return groups, majority

def offline_filter_maj(input_file, output_file, l_maj_c, r_maj_c):
    data_list = []
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            preds = entry['preds_group_idx']
            groups, maj_group_idx = group_pred(preds)
            if l_maj_c <= len(groups[maj_group_idx]) <= r_maj_c:
                data_list.append(
                    # {
                    #     "idx": entry["idx"],
                    #     "problem": entry["problem"],
                    #     "answer": entry["answer"],
                    #     "solution": entry.get("solution", ""),
                    # }
                    entry
                )

    # Save the preprocessed data to a new JSON file
    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + '\n')
    exit() 

def sample_data(data_path, output_path, sample_num):
    """
    Each data has a 'level' field; extract an equal number of data samples from each level.
    """
    # set seed
    random.seed(42)
    level2data = {}
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if not item['level'] in level2data:
                level2data[item['level']] = []
            level2data[item['level']].append(item)
    # Extract sample_num samples from each level in equal amounts.
    all_data_size = sum([len(level2data[level]) for level in level2data])
    print(f"all_data_size: {all_data_size}")

    each_level_need_sample_size = {}
    for level in level2data:
        if len(level2data[level]) < sample_num // len(level2data):
            each_level_need_sample_size[level] = len(level2data[level])
        else:
            each_level_need_sample_size[level] = sample_num // len(level2data)

    need_complement_sample_size = sample_num - sum([each_level_need_sample_size[level] for level in each_level_need_sample_size])

    while need_complement_sample_size > 0:
        for level in level2data:
            if len(level2data[level]) <= each_level_need_sample_size[level]:
                continue
            each_level_need_sample_size[level] += 1
            need_complement_sample_size -= 1
            if need_complement_sample_size == 0:
                break
    print(f"each_level_need_sample_size: {each_level_need_sample_size}")
    sample_level2data = {}
    for level in level2data:
        if len(level2data[level]) <= each_level_need_sample_size[level]:
            sample_level2data[level] = level2data[level]
        else:
            sample_level2data[level] = random.sample(level2data[level], each_level_need_sample_size[level])
    # output
    with open(output_path, 'w') as f:
        for level in sample_level2data:
            for item in sample_level2data[level]:
                f.write(json.dumps(item) + '\n')
    print(f"sampled data size: {[len(sample_level2data[level]) for level in sample_level2data]}")
    exit()

def sample_med_qa_data(data_path, output_path, sample_num):
    """
    从medqa数据文件中提取指定数量的数据样本。
    从step1题目中随机提取sample_num/2数量，从step2&3随机提取sample_num/2数量。
    
    Args:
        data_path: 输入的medqa jsonl文件路径
        output_path: 输出文件路径
        sample_num: 总采样数量
    """
    # 设置随机种子
    random.seed(42)
    
    # 按meta_info分类存储数据
    step1_data = []
    step2_3_data = []
    
    # 读取数据并分类
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                meta_info = item.get('meta_info', '')
                if meta_info == 'step1':
                    step1_data.append(item)
                elif meta_info == 'step2&3':
                    step2_3_data.append(item)
    
    print(f"总数据量: step1={len(step1_data)}, step2&3={len(step2_3_data)}")
    
    # 计算每类需要采样的数量
    each_type_sample_size = sample_num // 2
    
    # 从每类中随机采样
    sampled_step1 = random.sample(step1_data, min(each_type_sample_size, len(step1_data)))
    sampled_step2_3 = random.sample(step2_3_data, min(each_type_sample_size, len(step2_3_data)))
    
    # 如果某一类数据不足，从另一类补充
    total_sampled = len(sampled_step1) + len(sampled_step2_3)
    if total_sampled < sample_num:
        remaining_needed = sample_num - total_sampled
        if len(sampled_step1) < each_type_sample_size and len(step2_3_data) > len(sampled_step2_3):
            # step1不足，从step2&3补充
            remaining_step2_3 = [item for item in step2_3_data if item not in sampled_step2_3]
            additional_step2_3 = random.sample(remaining_step2_3, min(remaining_needed, len(remaining_step2_3)))
            sampled_step2_3.extend(additional_step2_3)
        elif len(sampled_step2_3) < each_type_sample_size and len(step1_data) > len(sampled_step1):
            # step2&3不足，从step1补充
            remaining_step1 = [item for item in step1_data if item not in sampled_step1]
            additional_step1 = random.sample(remaining_step1, min(remaining_needed, len(remaining_step1)))
            sampled_step1.extend(additional_step1)
    
    # 合并采样结果
    all_sampled_data = sampled_step1 + sampled_step2_3
    
    # 随机打乱顺序
    random.shuffle(all_sampled_data)
    
    # 输出到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"采样完成: step1={len(sampled_step1)}, step2&3={len(sampled_step2_3)}, 总计={len(all_sampled_data)}")
    print(f"输出文件: {output_path}")
    return len(all_sampled_data)

def merge_files(input_files: list[str], output_file: str):
    """
    Merge multiple JSON files into one.
    """
    with open(output_file, 'w') as outfile:
        for input_file in input_files:
            with open(input_file, 'r') as infile:
                for line in infile:
                    outfile.write(line)
    print(f"Merged {len(input_files)} files into {output_file}")
    exit()

def compare_online_offline_filtered_data(all_idxs, offline_keep_file, online_filtered_file, output_file):
    offline_keep_data_idxs = []
    with open(offline_keep_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            offline_keep_data_idxs.append(entry['idx'])

    # set(all_idxs)-set(offline_keep_data_idxs)
    offline_filtered_data_idxs = list(set(all_idxs) - set(offline_keep_data_idxs))

    online_filtered_data_idxs = []
    with open(online_filtered_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            online_filtered_data_idxs.append(entry['idx'])

    with open(output_file, 'w') as f:
        f.write('online total filtered data size: ' + str(len(online_filtered_data_idxs)) + '\n')
        f.write('offline total filtered data size: ' + str(len(offline_filtered_data_idxs)) + '\n')
        online_duo_idxs = list(set(online_filtered_data_idxs) - set(offline_filtered_data_idxs))
        f.write(f'Indices where online filtered data has more entries than offline filtered data ({len(online_duo_idxs)}):\n')

        for idx in online_duo_idxs:
            f.write(str(idx) + '\n')
        offline_duo_idxs = list(set(offline_filtered_data_idxs) - set(online_filtered_data_idxs))
        f.write(f'Indices where offline filtered data has more entries than online filtered data ({len(offline_duo_idxs)}):\n')
        for idx in offline_duo_idxs:
            f.write(str(idx) + '\n')

    exit()

def maj_acc(input_file, n):
    # Create the verification function
    verify_func = math_metric(
        # gold_extraction_target=(LatexExtractionConfig() if gold_is_latex else ExprExtractionConfig(),),
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=6
    )
    total_num = 0
    correct_num = 0
    with open(input_file, 'r') as f:
        for line in f:
            total_num += 1
            entry = json.loads(line)
            preds = entry['preds_group_idx']
            preds = preds[:n]
            groups, maj_group_idx = group_pred(preds)
            maj_answer = entry['code'][groups[maj_group_idx][0]]
            grade = 0
            try:
                grade, _ = verify_func([entry['answer']], [maj_answer])
                if grade != 1:
                    grade, _ = verify_func([maj_answer], [entry['answer']])
            except:
                grade = 0
            if grade == 1:
                correct_num += 1
    print(f"input_file: {input_file}")
    print(f"total_num: {total_num}, correct_num: {correct_num}, maj@{n}_acc: {correct_num / total_num}")
                
def print_eval_data_acc(math_eval_dir: Path, data_subdirs):
    data2acc = {}
    for data_subdir in data_subdirs:
        data_dir = math_eval_dir / data_subdir
        with open(data_dir / 'test_pure_-1_seed0_t0.0_s0_e-1_output_metrics.json', 'r') as f:
            data = json.load(f)
            data2acc[data_subdir] = data['accuracy']
    print(f"data2acc: {data2acc}")

def mmlu_pro_acc(input_file):
    # Match the number 0.3812 from "Average accuracy 0.3812"
    regex = re.compile(r'Average accuracy (\d+\.\d+)')
    mmlu_each_cat_num = [
        717, # 0 biology
        789, # 1 business
        1132, # 2 chemistry
        410, # 3 computer science
        844, # 4 economics
        969, # 5 engineering
        818, # 6 health
        381, # 7 history
        1101, # 8 law
        1351, # 9 math
        924, # 10 other
        499, # 11 philosophy
        1299, # 12 physics
        798, # 13 phychhology
    ]
    stem_idxs = [0, 2, 3, 5, 9, 12]
    humanities_idxs = [7, 8, 11]
    social_idxs = [1, 4, 6, 13]
    other_idxs = [10]
    with open(input_file, 'r') as f:
        data = f.read()

    # match all
    match = regex.findall(data)
    if match:
        match = [float(i) for i in match]
        stem_acc = 0
        humanities_acc = 0
        social_acc = 0
        other_acc = 0

        total = 0
        for idx in stem_idxs:
            stem_acc += match[idx]*mmlu_each_cat_num[idx]
            total += mmlu_each_cat_num[idx]
        stem_acc /= total
        print(f"stem_acc: {stem_acc}")

        total = 0
        for idx in humanities_idxs:
            humanities_acc += match[idx]*mmlu_each_cat_num[idx]
            total += mmlu_each_cat_num[idx]
        humanities_acc /= total
        print(f"humanities_acc: {humanities_acc}")

        total = 0
        for idx in social_idxs:
            social_acc += match[idx]*mmlu_each_cat_num[idx]
            total += mmlu_each_cat_num[idx]
        social_acc /= total
        print(f"social_acc: {social_acc}")

        total = 0
        for idx in other_idxs:
            other_acc += match[idx]*mmlu_each_cat_num[idx]
            total += mmlu_each_cat_num[idx]
        other_acc /= total
        print(f"other_acc: {other_acc}")

        avg_acc = 0
        total = 0
        for idx in range(len(match)):
            avg_acc += match[idx]*mmlu_each_cat_num[idx]
            total += mmlu_each_cat_num[idx]
        avg_acc /= total
        print(f"avg_acc: {avg_acc}")
    else:
        print("No match found")
    regex = re.compile(r'Average accuracy: (\d+\.\d+)')
    match = regex.search(data)
    if match:
        accuracy = match.group(1)
        print(f"Average accuracy: {accuracy}")
    else:
        print("No match found")



def extract_correct_answers(input_file, output_file):
    """
    从examples.jsonl文件中提取每个问题的正确答案。
    根据code_evaluations中的is_correct字段判断哪个回答是正确的，
    如果所有回答都错误则使用solution字段。
    
    Args:
        input_file: 输入的examples.jsonl文件路径
        output_file: 输出的jsonl文件路径，包含instruction、input、output字段
    """
    data_list = []
    
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            
            # 获取问题
            question = entry.get("question", "")
            if not question:
                continue
                
            # 查找第一个正确的回答
            correct_answer = None
            code_evaluations = entry.get("code_evaluations", [])
            code_responses = entry.get("code", [])
            
            # 遍历评估结果，找到第一个正确的回答
            for eval_item in code_evaluations:
                if eval_item.get("is_correct", False):
                    response_index = eval_item.get("response_index", -1)
                    if response_index == -1:
                        continue
                    if response_index < len(code_responses):
                        correct_answer = code_responses[response_index]
                        break
            
            # 如果没有找到正确答案，使用solution字段
            if correct_answer is None:
                correct_answer = entry.get("solution", "")
            
            # 构建输出数据
            if correct_answer:
                data_item = {
                    "instruction": question,
                    "input": "",
                    "output": correct_answer
                }
                data_list.append(data_item)
    
    # 保存到输出文件
    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"提取了 {len(data_list)} 个正确答案，保存到 {output_file}")
    exit()


def merge_maj_reward_to_rule_self_reward(maj_file, rule_self_file, output_file):
    # maj_data = []
    idx2majdata = {}
    with open(maj_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # maj_data.append(entry)
            idx2majdata[entry['idx']] = entry

    merged_data = []
    with open(rule_self_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            maj_item = idx2majdata.get(entry['idx'])
            entry['maj_num'] = maj_item['maj_num']
            entry['preds_group_idx'] = maj_item['preds_group_idx']
            entry['total'] = maj_item['total']
            merged_data.append(entry)

    # assert len(maj_data) == len(rule_self_data), "maj data and rule self data length not equal"

    with open(output_file, 'w') as f:
        for entry in merged_data:
            f.write(json.dumps(entry) + '\n')

    print(f"merged data size: {len(merged_data)}")
    exit()

if __name__ == "__main__":
    input_file = "/pubshare/fwk/code/SeRL/evaluation/Health/dataset/med_qa/train.jsonl"
    output_file = "/pubshare/fwk/code/SeRL/evaluation/Health/dataset/med_qa/train_500seed.jsonl"
    sample_num = 500
    sample_med_qa_data(input_file, output_file, sample_num)

    # maj_file = "/xxx/SEO/Math-Verify/outputs/xxx/Qwen/Qwen2.5-7B-Instruct/math_eval_bon_16/math_500/qwen25_7b_maj_eval.jsonl"
    # rule_self_file = "/xxx/SEO/Math-Verify/outputs/xxx/Qwen/Qwen2.5-7B-Instruct/math_eval_bon_16/math_500/test_pure_-1_seed0_t1.0_s0_e-1_with_self_math_reward_with_rule_reward.jsonl"
    # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/Qwen/Qwen2.5-7B-Instruct/math_eval_bon_16/math_500/qwen25_7b_math500_with_self_math_reward_with_rule_reward_maj_eval.jsonl"
    # merge_maj_reward_to_rule_self_reward(
    #     maj_file=maj_file,
    #     rule_self_file=rule_self_file,
    #     output_file=output_file
    # )

    # input_file = "/xxx/SEO/MMLU-Pro/eval_results/summary/qwen25_7B-origin_7_5k_random_bon_gt_bs16-CoT-all_05-02_03-35_summary.txt"
    # mmlu_pro_acc(input_file)

    # math_eval_dir = Path("/xxx/SEO/Math-Verify/outputs/xxx/orlhf_checkpoints/llama32_3B-random_bon_maj_bs16_seo_rloo2/global_step100_hf/math_eval")
    # data_subdirs = [
    #     'math_500', 'math_hard', 'asdiv', 'college_math', 'tabmwp'
    #     # 'asdiv', 'carp_en', 'college_math', 'gaokao2023en', 'mawps', 
    #     # 'minerva_math', 'mmlu_stem', 'olympiadbench', 'svamp', 'tabmwp'
    # ]
    # print_eval_data_acc(math_eval_dir, data_subdirs)

    # input_files = [
    #     "/xxx/SEO/Math-Verify/outputs/xxx/Qwen/Qwen2.5-7B-Instruct/math_eval_bon_32/gsm8k/test_pure_-1_seed0_t1.0_s0_e-1_maj_eval.jsonl",
    #     "/xxx/SEO/Math-Verify/outputs/xxx/Qwen/Qwen2.5-7B-Instruct/math_eval_bon_32/math_500/test_pure_-1_seed0_t1.0_s0_e-1_maj_eval.jsonl"
    # ]
    # for input_file in input_files:
    #     for n in [16, 32]:
    #         maj_acc(input_file, n)

    # # # file_path = '/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/0208_500seed_gen7500_pure_-1_seed0_t1.0_s0_e-1_maj_eval_filtered0208_with_self_rewards.jsonl'
    # # file_path = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1_with_rule_rewards.jsonl"
    # # file_path = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/0208_500seed_gen7500_pure_-1_seed0_t1.0_s0_e-1_maj_eval_filtered0208_with_self_math_rewards.jsonl"
    # # file_path = "/xxx/SEO/Math-Verify/outputs/xxx/checkpoints/dpo/llama32_3B_rule_rewards_dpo_iter0/checkpoint-1400/math_eval_bon_4/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1_with_rule_rewards.jsonl"
    # # file_path = "/xxx/SEO/Math-Verify/outputs/xxx/checkpoints/dpo/llama32_3B_rule_rewards_dpo_iter1/checkpoint-600/math_eval_bon_4/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1_with_rule_rewards.jsonl"
    # file_path = "/xxx/SEO/Math-Verify/outputs/xxx/checkpoints/dpo/llama32_3B_self_math_rewards_dpo_iter0/checkpoint-372/math_eval_bon_4/math/self_math_rewards_0208_500seed_gen7500_iter1_pure_-1_seed0_t1.0_s0_e-1_with_self_math_rewards.jsonl"
    # reward_key = "self_reward_rewards"
    # # reward_key = 'rule_rewards'
    # # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/0208_500seed_gen7500_maj_eval_filtered0208_self_rewards_dpo_dataset_iter0.jsonl"
    # # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/math_train_rule_rewards_dpo_dataset_iter0.jsonl"
    # # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/0208_500seed_gen7500_maj_eval_filtered0208_self_math_rewards_dpo_dataset_iter0.jsonl"
    # # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/checkpoints/dpo/llama32_3B_rule_rewards_dpo_iter0/checkpoint-1400/math_eval_bon_4/math/math_train_rule_rewards_dpo_dataset_iter1.jsonl"
    # # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/checkpoints/dpo/llama32_3B_rule_rewards_dpo_iter1/checkpoint-600/math_eval_bon_4/math/math_train_rule_rewards_dpo_dataset_iter2.jsonl"
    # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/checkpoints/dpo/llama32_3B_self_math_rewards_dpo_iter0/checkpoint-372/math_eval_bon_4/math/0208_500seed_gen7500_self_math_rewards_dpo_dataset_iter1.jsonl"
    # create_dpo_dataset(
    #     input_file=file_path,
    #     reward_key=reward_key,
    #     output_file=output_file
    # )

    # file_path = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1_with_rule_rewards.jsonl"
    # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/math_train_rule_rewards_rejection_sft_iter0.jsonl"
    # reward_key = "rule_rewards"
    # create_sft_dataset(
    #     input_file=file_path,
    #     reward_key=reward_key,
    #     output_file=output_file
    # )

    # input_file = "/xxx/SEO/Math-Verify/outputs/xxx/orlhf_checkpoints/checkpoint/llama3-3b-0208seed_gen7500_maj_filtered_0208_random_bon_maj_bs16/global_step200_hf/math_eval_bon_32/math/seo_0208_500seed_gen7500_iter1_pure_-1_seed0_t1.0_s0_e-1_maj_eval.jsonl"
    # # input_file = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/0208_500seed_gen7500_pure_-1_seed0_t1.0_s0_e-1_maj_eval_with_self_rewards.jsonl"
    # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/orlhf_checkpoints/checkpoint/llama3-3b-0208seed_gen7500_maj_filtered_0208_random_bon_maj_bs16/global_step200_hf/math_eval_bon_32/math/seo_0208_500seed_gen7500_iter1_pure_-1_seed0_t1.0_s0_e-1_maj_eval_filtered_0208.jsonl"
    # # output_file = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/0208_500seed_gen7500_pure_-1_seed0_t1.0_s0_e-1_maj_eval_filtered0208_with_self_rewards.jsonl"
    # l_maj_c = 32*0.2
    # r_maj_c = 32*0.8
    # offline_filter_maj(
    #     input_file=input_file,
    #     output_file=output_file,
    #     l_maj_c=l_maj_c,
    #     r_maj_c=r_maj_c
    # )

    # input_files = [
    #     # "/xxx/SEO/self-instruct/data/llama3_2_3b_generations/seo_0208_500seed_gen7500_iter1/2025-04-25_03-06-44/machine_generated_instructions_4090.jsonl",
    #     # "/xxx/SEO/self-instruct/data/llama3_2_3b_generations/seo_0208_500seed_gen7500_iter1/2025-04-25_03-06-44/machine_generated_instructions_a6000.jsonl"
    #     f"/xxx/SEO/MCGEP/train_online_filtered_data_dir/llama3-3b-0208seed_gen7500_online_filtered_0208_random_bon_maj_bs16/2025-04-26_11-58-23/filtered_data_{i}.jsonl" for i in range(1, 203)
    # ]
    # # output_file = "/xxx/SEO/self-instruct/data/llama3_2_3b_generations/seo_0208_500seed_gen7500_iter1/2025-04-25_03-06-44/machine_generated_instructions_a6000_merged.jsonl"
    # output_file = "/xxx/SEO/MCGEP/train_online_filtered_data_dir/llama3-3b-0208seed_gen7500_online_filtered_0208_random_bon_maj_bs16/2025-04-26_11-58-23/filtered_data_merged.jsonl"
    # merge_files(
    #     input_files=input_files,
    #     output_file=output_file
    # )
    # offline_keep_file = "/xxx/SEO/Math-Verify/outputs/xxx/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/0208_500seed_gen7500_pure_-1_seed0_t1.0_s0_e-1_maj_eval_filtered0208_with_self_rewards.jsonl"
    # online_filtered_file = "/xxx/SEO/MCGEP/train_online_filtered_data_dir/llama3-3b-0208seed_gen7500_online_filtered_0208_random_bon_maj_bs16/2025-04-26_11-58-23/filtered_data_merged.jsonl"
    # all_idxs = [i for i in range(7507)]
    # output_file = "/xxx/SEO/MCGEP/train_online_filtered_data_dir/llama3-3b-0208seed_gen7500_online_filtered_0208_random_bon_maj_bs16/2025-04-26_11-58-23/online_offline_filtered_data_compare.txt"
    # compare_online_offline_filtered_data(
    #     all_idxs=all_idxs,
    #     offline_keep_file=offline_keep_file,
    #     online_filtered_file=online_filtered_file,
    #     output_file=output_file
    # )

    
