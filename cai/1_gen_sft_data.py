
import json
import random
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import threading
from functools import partial

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 格式的数据文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存数据到 JSONL 格式文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_critiques(file_path: str) -> Dict[str, Dict[str, str]]:
    """加载批评提示模板"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_few_shot_examples(file_path: str) -> str:
    """加载少样本学习示例"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def format_problem_prompt(problem: str, few_shot_examples: str) -> str:
    """格式化问题提示"""
    return f"{few_shot_examples}\n\nHuman: {problem}\n\nAssistant:"

def format_critique_prompt(problem: str, response: str, critique_request: str, few_shot_examples: str) -> str:
    """格式化批评提示"""
    return f"{few_shot_examples}\n\nHuman: {problem}\n\nAssistant: {response}\n\nCritiqueRequest: {critique_request}\n\nCritique:"

def format_revision_prompt(problem: str, response: str, critique: str, edit_request: str, few_shot_examples: str, critique_request: str) -> str:
    """格式化修订提示"""
    return f"{few_shot_examples}\n\nHuman: {problem}\n\nAssistant: {response}\n\nCritiqueRequest: {critique_request}\n\nCritique: {critique}\n\nRevisionRequest: {edit_request}\n\nRevision:"

def generate_response(llm: LLM, prompt: str, sampling_params: SamplingParams) -> str:
    """使用 vLLM 生成回答"""
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()

def split_data(data: List[Dict[str, Any]], num_chunks: int) -> List[List[Dict[str, Any]]]:
    """将数据分割成多个块用于多进程处理"""
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    
    chunks = []
    start = 0
    for i in range(num_chunks):
        # 前 remainder 个块多分配一个元素
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_chunk_size
        chunks.append(data[start:end])
        start = end
    
    return chunks

def worker_process(args_tuple):
    """工作进程函数"""
    (
        process_id,
        data_chunk,
        model_path,
        critiques_path,
        few_shot_path,
        temperature,
        max_tokens,
        tensor_parallel_size,
        gpu_id,
        seed
    ) = args_tuple
    
    try:
        # 设置 GPU 设备
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 设置随机种子（每个进程使用不同的种子）
        random.seed(seed + process_id)
        
        print(f"进程 {process_id} 开始加载模型，使用 GPU {gpu_id}...")
        
        # 加载模型
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True
        )
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n\nH:", "\n\nHuman:", "\n\n-----", "CritiqueRequest", "\n\nCritiqueRequest", "Critique", "\n\nCritique", "RevisionRequest", "\n\nRevisionRequest", "Revision", "\n\nRevision"],
        )
        
        # 加载提示模板
        critiques = load_critiques(critiques_path)
        few_shot_examples = load_few_shot_examples(few_shot_path)
        
        print(f"进程 {process_id} 开始处理 {len(data_chunk)} 个问题...")
        
        # 处理数据块
        results = []
        for i, problem_data in enumerate(data_chunk):
            try:
                result = process_single_problem(
                    llm=llm,
                    problem_data=problem_data,
                    critiques=critiques,
                    few_shot_examples=few_shot_examples,
                    sampling_params=sampling_params
                )
                results.append(result)
                
                # 每处理 5 个问题打印一次进度
                if (i + 1) % 5 == 0:
                    print(f"进程 {process_id}: 已处理 {i + 1}/{len(data_chunk)} 个问题")
                    
            except Exception as e:
                print(f"进程 {process_id} 处理第 {i+1} 个问题时出错: {e}")
                continue
        
        print(f"进程 {process_id} 完成，成功处理 {len(results)}/{len(data_chunk)} 个问题")
        return process_id, results
        
    except Exception as e:
        print(f"进程 {process_id} 发生严重错误: {e}")
        return process_id, []

def process_single_problem(llm: LLM, problem_data: Dict[str, Any], critiques: Dict[str, Dict[str, str]], 
                          few_shot_examples: str, sampling_params: SamplingParams) -> Dict[str, Any]:
    """处理单个数学问题：生成初始回答 -> 批评 -> 修订"""
    problem = problem_data['problem']
    
    # 1. 生成初始回答
    initial_prompt = format_problem_prompt(problem, few_shot_examples)
    initial_response = generate_response(llm, initial_prompt, sampling_params)
    
    # 2. 随机选择一个批评类型
    critique_id = random.choice(list(critiques.keys()))
    critique_request = critiques[critique_id]['critique_request']
    edit_request = critiques[critique_id]['edit_request']
    
    # 3. 生成批评
    critique_prompt = format_critique_prompt(problem, initial_response, critique_request, few_shot_examples)
    critique = generate_response(llm, critique_prompt, sampling_params)
    
    # 4. 基于批评生成修订回答
    revision_prompt = format_revision_prompt(problem, initial_response, critique, edit_request, few_shot_examples, critique_request)
    revised_response = generate_response(llm, revision_prompt, sampling_params)
    
    # 5. 返回结果
    result = {
        'problem': problem,
        'answer': problem_data.get('answer', ''),
        'subject': problem_data.get('subject', ''),
        'level': problem_data.get('level', ''),
        'unique_id': problem_data.get('unique_id', ''),
        'initial_response': initial_response,
        'critique_id': critique_id,
        'critique_request': critique_request,
        'critique': critique,
        'edit_request': edit_request,
        'revised_response': revised_response
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='生成 SFT 数据：初始回答 -> 批评 -> 修订')
    parser.add_argument('--model_path', type=str, required=True, help='vLLM 模型路径')
    parser.add_argument('--data_path', type=str, required=True, help='数学数据集 JSONL 文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出文件路径')
    parser.add_argument('--critiques_path', type=str, default='prompts/critiques.json', help='批评提示文件路径')
    parser.add_argument('--few_shot_path', type=str, default='prompts/few_shot.txt', help='少样本示例文件路径')
    parser.add_argument('--max_samples', type=int, default=None, help='最大处理样本数量')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--max_tokens', type=int, default=2048, help='最大生成 token 数')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='张量并行大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 多进程相关参数
    parser.add_argument('--num_processes', type=int, default=1, help='并行进程数量')
    parser.add_argument('--gpu_devices', type=str, default='0', help='使用的 GPU 设备列表，用逗号分隔，如 "0,1,2,3"')
    parser.add_argument('--use_multiprocessing', action='store_true', help='是否启用多进程并行')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    print("正在加载数据...")
    # 加载数学数据集
    math_data = load_jsonl(args.data_path)
    if args.max_samples:
        math_data = math_data[1000:1000+args.max_samples]
    
    print(f"总共需要处理 {len(math_data)} 个问题")
    
    if args.use_multiprocessing and args.num_processes > 1:
        # 多进程模式
        print(f"启用多进程模式，使用 {args.num_processes} 个进程")
        
        # 解析 GPU 设备列表
        gpu_devices = [int(x.strip()) for x in args.gpu_devices.split(',')]
        print(f"可用 GPU 设备: {gpu_devices}")
        
        # 确保进程数不超过 GPU 数量
        if args.num_processes > len(gpu_devices):
            print(f"警告：进程数 ({args.num_processes}) 超过 GPU 数量 ({len(gpu_devices)})，将进程数调整为 GPU 数量")
            args.num_processes = len(gpu_devices)
        
        # 分割数据
        data_chunks = split_data(math_data, args.num_processes)
        print(f"数据已分割为 {len(data_chunks)} 个块，每块大小: {[len(chunk) for chunk in data_chunks]}")
        
        # 准备进程参数
        process_args = []
        for i in range(args.num_processes):
            gpu_id = gpu_devices[i % len(gpu_devices)]  # 循环分配 GPU
            process_args.append((
                i,  # process_id
                data_chunks[i],  # data_chunk
                args.model_path,
                args.critiques_path,
                args.few_shot_path,
                args.temperature,
                args.max_tokens,
                args.tensor_parallel_size,
                gpu_id,
                args.seed
            ))
        
        # 启动多进程处理
        start_time = time.time()
        all_results = []
        
        print("开始多进程处理...")
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            # 提交所有任务
            future_to_process = {executor.submit(worker_process, arg): i for i, arg in enumerate(process_args)}
            
            # 收集结果
            for future in as_completed(future_to_process):
                process_id = future_to_process[future]
                try:
                    returned_process_id, results = future.result()
                    all_results.extend(results)
                    print(f"进程 {returned_process_id} 完成，返回 {len(results)} 个结果")
                except Exception as e:
                    print(f"进程 {process_id} 执行失败: {e}")
        
        end_time = time.time()
        print(f"多进程处理完成，总耗时: {end_time - start_time:.2f} 秒")
        results = all_results
        
    else:
        # 单进程模式（原有逻辑）
        print("使用单进程模式")
        
        print("正在加载模型...")
        # 加载 vLLM 模型
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True
        )
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["\n\nH:", "\n\nHuman:", "\n\n-----"],
        )
        
        # 加载批评提示模板
        critiques = load_critiques(args.critiques_path)
        
        # 加载少样本学习示例
        few_shot_examples = load_few_shot_examples(args.few_shot_path)
        
        print(f"开始处理 {len(math_data)} 个问题...")
        
        # 逐一处理每个问题
        results = []
        for i, problem_data in enumerate(tqdm(math_data, desc="处理问题")):
            try:
                result = process_single_problem(
                    llm=llm,
                    problem_data=problem_data,
                    critiques=critiques,
                    few_shot_examples=few_shot_examples,
                    sampling_params=sampling_params
                )
                results.append(result)
                
                # 每处理 10 个问题保存一次（防止数据丢失）
                if (i + 1) % 10 == 0:
                    save_jsonl(results, args.output_path)
                    print(f"已处理 {i + 1}/{len(math_data)} 个问题，中间结果已保存")
                    
            except Exception as e:
                print(f"处理第 {i+1} 个问题时出错: {e}")
                continue
    
    # 保存最终结果
    save_jsonl(results, args.output_path)
    print(f"处理完成！共生成 {len(results)} 个样本，结果已保存到 {args.output_path}")
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"总问题数: {len(math_data)}")
    print(f"成功处理: {len(results)}")
    print(f"成功率: {len(results)/len(math_data)*100:.1f}%")
    
    # 统计批评类型分布
    critique_counts = {}
    for result in results:
        critique_id = result['critique_id']
        critique_counts[critique_id] = critique_counts.get(critique_id, 0) + 1
    
    print("\n批评类型分布:")
    for critique_id, count in sorted(critique_counts.items()):
        print(f"  类型 {critique_id}: {count} 次")

if __name__ == '__main__':
    main()
