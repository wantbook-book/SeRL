#!/usr/bin/env python3
"""
医疗问答数据集评估脚本
用于评估模型生成的答案与标准答案的准确率
"""

import json
import re
import argparse
from typing import Dict, List, Tuple
from pathlib import Path


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def detect_dataset_type(data: List[Dict]) -> str:
    """
    检测数据集类型
    根据数据中的字段和答案格式判断是MedQA/NephSAP还是PubMedQA
    """
    if not data:
        return "unknown"
    
    # 检查第一个样本
    sample = data[0]
    
    # 如果有extracted_answer字段且值为yes/no/maybe，则为PubMedQA
    if 'extracted_answer' in sample:
        extracted = sample['extracted_answer'].lower()
        if extracted in ['yes', 'no', 'maybe']:
            return "pubmedqa"
    
    # 检查answer字段
    if 'answer' in sample:
        answer = str(sample['answer']).lower()
        if answer in ['yes', 'no', 'maybe']:
            return "pubmedqa"
    
    # 检查answer_idx字段
    if 'answer_idx' in sample:
        answer_idx = str(sample['answer_idx']).upper()
        if answer_idx in ['A', 'B', 'C', 'D', 'E']:
            return "medqa"  # MedQA和NephSAP都使用ABCDE格式
    
    # 默认返回medqa
    return "medqa"


def extract_multiple_choice_answer(response: str) -> str:
    """
    从生成的回答中提取多选题答案选项 (A, B, C, D, E)
    用于MedQA和NephSAP数据集
    """
    if not response:
        return ""
    
    # 如果response是列表，取第一个元素
    if isinstance(response, list) and len(response) > 0:
        response = response[0]
    
    # 转换为字符串
    response = str(response)
    
    # 1. 查找 \boxed{} 格式
    boxed_pattern = r'\\boxed\{([A-E])\}'
    boxed_match = re.search(boxed_pattern, response, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()
    
    # 2. 查找 "The answer is X" 格式
    answer_patterns = [
        r'(?:the\s+)?(?:best\s+)?(?:correct\s+)?answer\s+is\s+([A-E])',
        r'(?:答案|正确答案)(?:是|为)\s*([A-E])',
        r'选择\s*([A-E])',
        r'选项\s*([A-E])'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 3. 查找最后出现的单独选项字母
    # single_letter_pattern = r'\b([A-E])\b'
    # matches = re.findall(single_letter_pattern, response, re.IGNORECASE)
    # if matches:
    #     return matches[-1].upper()  # 返回最后一个匹配的选项
    
    # 4. 如果都没找到，返回空字符串
    return ""


def extract_pubmedqa_answer(response: str) -> str:
    """
    从生成的回答中提取PubMedQA答案 (yes, no, maybe)
    用于PubMedQA数据集
    """
    if not response:
        return ""
    
    # 如果response是列表，取第一个元素
    if isinstance(response, list) and len(response) > 0:
        response = response[0]
    
    # 转换为字符串
    response = str(response).lower()
    
    # 1. 查找 \boxed{} 格式
    boxed_pattern = r'\\boxed\{(yes|no|maybe)\}'
    boxed_match = re.search(boxed_pattern, response, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).lower()
    
    # 2. 查找明确的答案模式
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s+(yes|no|maybe)',
        r'(?:my\s+)?(?:final\s+)?(?:answer|conclusion)\s*:?\s*(yes|no|maybe)',
        r'\b(yes|no|maybe)\b(?:\.|$|\s*$)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # 3. 基于关键词的启发式匹配
    # if 'yes' in response and 'no' not in response:
    #     return 'yes'
    # elif 'no' in response and 'yes' not in response:
    #     return 'no'
    # elif any(word in response for word in ['maybe', 'uncertain', 'unclear', 'possible']):
    #     return 'maybe'
    
    # 4. 默认返回maybe
    return ''


def extract_answer_from_response(response: str, dataset_type: str = "medqa") -> str:
    """
    根据数据集类型从生成的回答中提取答案
    """
    if dataset_type == "pubmedqa":
        return extract_pubmedqa_answer(response)
    else:
        # MedQA和NephSAP都使用多选题格式
        return extract_multiple_choice_answer(response)


def evaluate_accuracy(data: List[Dict]) -> Tuple[float, Dict, List[Dict]]:
    """
    计算准确率
    返回: (总准确率, 详细统计信息, 增强的数据列表)
    """
    total_count = len(data)
    correct_count = 0
    no_answer_count = 0
    
    # 检测数据集类型
    dataset_type = detect_dataset_type(data)
    print(f"检测到数据集类型: {dataset_type}")
    
    # 按meta_info分类统计
    meta_stats = {}
    
    # 增强的数据列表，添加提取的答案和正确性字段
    enhanced_data = []
    
    for item in data:
        # 根据数据集类型获取正确答案
        if dataset_type == "pubmedqa":
            # PubMedQA使用answer字段，值为yes/no/maybe
            correct_answer = str(item.get('answer', '')).lower()
        else:
            # MedQA和NephSAP使用answer_idx字段，值为A/B/C/D/E
            correct_answer = str(item.get('answer_idx', '')).upper()
        
        generated_response = item.get('generated_response', '')
        predicted_answer = extract_answer_from_response(generated_response, dataset_type)
        meta_info = item.get('meta_info', 'unknown')
        
        # 判断是否正确
        is_correct = False
        if predicted_answer and predicted_answer == correct_answer:
            is_correct = True
            correct_count += 1
        elif not predicted_answer:
            no_answer_count += 1
        
        # 创建增强的数据项
        enhanced_item = item.copy()
        enhanced_item['extracted_answer'] = predicted_answer
        enhanced_item['is_correct'] = is_correct
        enhanced_item['dataset_type'] = dataset_type
        enhanced_data.append(enhanced_item)
        
        # 初始化meta_info统计
        if meta_info not in meta_stats:
            meta_stats[meta_info] = {
                'total': 0,
                'correct': 0,
                'no_answer': 0
            }
        
        meta_stats[meta_info]['total'] += 1
        
        if not predicted_answer:
            meta_stats[meta_info]['no_answer'] += 1
        elif is_correct:
            meta_stats[meta_info]['correct'] += 1
    
    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # 计算各meta_info的准确率
    for meta_info in meta_stats:
        stats = meta_stats[meta_info]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    detailed_stats = {
        'total_count': total_count,
        'correct_count': correct_count,
        'no_answer_count': no_answer_count,
        'accuracy': accuracy,
        'dataset_type': dataset_type,
        'meta_stats': meta_stats
    }
    
    return accuracy, detailed_stats, enhanced_data


def print_evaluation_results(accuracy: float, stats: Dict):
    """打印评估结果"""
    dataset_type = stats.get('dataset_type', 'unknown')
    dataset_name_map = {
        'pubmedqa': 'PubMedQA',
        'medqa': 'MedQA/NephSAP',
        'unknown': '未知数据集'
    }
    dataset_name = dataset_name_map.get(dataset_type, dataset_type)
    
    print("=" * 60)
    print(f"医疗问答数据集评估结果 - {dataset_name}")
    print("=" * 60)
    print(f"数据集类型: {dataset_type}")
    if dataset_type == 'pubmedqa':
        print("答案格式: yes/no/maybe")
    else:
        print("答案格式: A/B/C/D/E")
    print(f"总题目数量: {stats['total_count']}")
    print(f"正确回答数量: {stats['correct_count']}")
    print(f"无法提取答案数量: {stats['no_answer_count']}")
    print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    print("按题目类型分类统计:")
    print("-" * 40)
    for meta_info, meta_stat in stats['meta_stats'].items():
        print(f"{meta_info}:")
        print(f"  题目数量: {meta_stat['total']}")
        print(f"  正确数量: {meta_stat['correct']}")
        print(f"  无答案数量: {meta_stat['no_answer']}")
        print(f"  准确率: {meta_stat['accuracy']:.4f} ({meta_stat['accuracy']*100:.2f}%)")
        print()


def save_evaluation_report(accuracy: float, stats: Dict, output_file: str):
    """保存评估报告到文件"""
    report = {
        'overall_accuracy': accuracy,
        'detailed_statistics': stats,
        'summary': {
            'total_questions': stats['total_count'],
            'correct_answers': stats['correct_count'],
            'accuracy_percentage': f"{accuracy*100:.2f}%"
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"评估报告已保存到: {output_file}")


def save_enhanced_data(enhanced_data: List[Dict], output_file: str):
    """保存增强的数据到JSONL文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in enhanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"增强数据已保存到: {output_file}")


def save_summary_json(accuracy: float, stats: Dict, output_file: str):
    """保存简洁的summary JSON文件"""
    summary = {
        'evaluation_summary': {
            'overall_accuracy': round(accuracy, 4),
            'accuracy_percentage': f"{accuracy*100:.2f}%",
            'total_questions': stats['total_count'],
            'correct_answers': stats['correct_count'],
            'incorrect_answers': stats['total_count'] - stats['correct_count'] - stats['no_answer_count'],
            'no_answer_count': stats['no_answer_count']
        },
        'by_category': {}
    }
    
    # 添加按类别的统计
    for meta_info, meta_stat in stats['meta_stats'].items():
        summary['by_category'][meta_info] = {
            'accuracy': round(meta_stat['accuracy'], 4),
            'accuracy_percentage': f"{meta_stat['accuracy']*100:.2f}%",
            'total': meta_stat['total'],
            'correct': meta_stat['correct'],
            'incorrect': meta_stat['total'] - meta_stat['correct'] - meta_stat['no_answer'],
            'no_answer': meta_stat['no_answer']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Summary报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='评估医疗问答数据集的准确率')
    parser.add_argument('input_file', help='输入的JSONL文件路径')
    parser.add_argument('--output', '-o', help='输出评估报告的JSON文件路径')
    parser.add_argument('--enhanced-output', help='输出增强数据的JSONL文件路径')
    parser.add_argument('--summary', '-s', help='输出summary JSON文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input_file).exists():
        print(f"错误: 输入文件不存在: {args.input_file}")
        return
    
    print(f"正在加载数据: {args.input_file}")
    data = load_jsonl(args.input_file)
    print(f"加载了 {len(data)} 条数据")
    
    if len(data) == 0:
        print("警告: 数据文件为空")
        return
    
    print("正在评估准确率...")
    accuracy, stats, enhanced_data = evaluate_accuracy(data)
    
    # 打印结果
    print_evaluation_results(accuracy, stats)
    
    # 自动生成输出文件名
    input_path = Path(args.input_file)
    base_name = input_path.stem
    output_dir = input_path.parent
    
    # 保存详细报告
    if args.output:
        save_evaluation_report(accuracy, stats, args.output)
    else:
        default_report_file = output_dir / f"{base_name}_evaluation_report.json"
        save_evaluation_report(accuracy, stats, str(default_report_file))
    
    # 保存增强数据
    if args.enhanced_output:
        save_enhanced_data(enhanced_data, args.enhanced_output)
    else:
        default_enhanced_file = output_dir / f"{base_name}_enhanced.jsonl"
        save_enhanced_data(enhanced_data, str(default_enhanced_file))
    
    # 保存summary
    if args.summary:
        save_summary_json(accuracy, stats, args.summary)
    else:
        default_summary_file = output_dir / f"{base_name}_summary.json"
        save_summary_json(accuracy, stats, str(default_summary_file))
    
    # 详细信息
    if args.verbose:
        print("\n详细错误分析:")
        print("-" * 40)
        error_count = 0
        dataset_type = stats.get('dataset_type', 'unknown')
        
        for i, item in enumerate(enhanced_data):
            if not item['is_correct']:
                error_count += 1
                if error_count <= 10:  # 只显示前10个错误
                    print(f"错误 {error_count} (索引 {i}):")
                    
                    # 根据数据集类型显示正确答案
                    if dataset_type == "pubmedqa":
                        correct_answer = item.get('answer', 'N/A')
                    else:
                        correct_answer = item.get('answer_idx', 'N/A')
                    
                    print(f"  正确答案: {correct_answer}")
                    print(f"  预测答案: {item['extracted_answer'] if item['extracted_answer'] else '无法提取'}")
                    print(f"  题目类型: {item.get('meta_info', 'unknown')}")
                    print(f"  数据集类型: {dataset_type}")
                    print()
        
        if error_count > 10:
            print(f"... 还有 {error_count - 10} 个错误未显示")


if __name__ == '__main__':
    main()