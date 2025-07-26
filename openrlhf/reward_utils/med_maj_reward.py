import re
import torch

def extract_answer_from_response(response: str) -> str:
    """
    从生成的回答中提取答案选项
    支持多种格式：
    1. \boxed{A}, \boxed{B} 等
    2. "The answer is A", "The best answer is B" 等
    3. "答案是A", "正确答案是B" 等
    4. 直接的选项字母 A, B, C, D, E
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
    # answer_patterns = [
    #     r'(?:the\s+)?(?:best\s+)?(?:correct\s+)?answer\s+is\s+([A-E])',
    #     r'(?:答案|正确答案)(?:是|为)\s*([A-E])',
    #     r'选择\s*([A-E])',
    #     r'选项\s*([A-E])'
    # ]
    
    # for pattern in answer_patterns:
    #     match = re.search(pattern, response, re.IGNORECASE)
    #     if match:
    #         return match.group(1).upper()
    
    # # 3. 查找最后出现的单独选项字母
    # single_letter_pattern = r'\b([A-E])\b'
    # matches = re.findall(single_letter_pattern, response, re.IGNORECASE)
    # if matches:
    #     return matches[-1].upper()  # 返回最后一个匹配的选项
    
    # 4. 如果都没找到，返回空字符串
    return ""

def reward_func(queries: list[str], prompts: list[str], labels: list[str]) -> torch.Tensor:
    """
    医学问答的多数投票奖励函数
    queries: list of strings, 每个字符串是一个问题和回答
    prompts: list of strings, 提示信息（在此实现中未使用）
    labels: list of strings, 正确答案标签（在此实现中未使用，因为使用多数投票）
    """
    
    # 从每个回答中提取答案
    extracted_answers = []
    for query in queries:
        answer = extract_answer_from_response(query)
        extracted_answers.append(answer)
    
    # 构建等价矩阵：比较每对答案是否相同
    equal_matrix = [[False] * len(queries) for _ in range(len(queries))]
    for i in range(len(queries)):
        for j in range(i, len(queries)):
            if i == j:
                equal_matrix[i][j] = True
                equal_matrix[j][i] = True
            else:
                # 比较提取的答案是否相同
                is_equal = False
                if extracted_answers[i] and extracted_answers[j]:
                    is_equal = extracted_answers[i] == extracted_answers[j]
                equal_matrix[i][j] = is_equal
                equal_matrix[j][i] = is_equal
    
    # 计算每个答案的支持数量（多数投票）
    maj_count = [0] * len(queries)
    for i in range(len(queries)):
        maj_count[i] = sum(equal_matrix[i])
    
    # 找到支持数量最多的答案索引（多数答案）
    majority_pred_idx = maj_count.index(max(maj_count))
    
    # 分配奖励：与多数答案相同的回答获得奖励1.0，否则为0.0
    reward_list = [0.0] * len(queries)
    for idx in range(len(queries)):
        if equal_matrix[majority_pred_idx][idx]:
            reward_list[idx] = 1.0
    
    return torch.tensor(reward_list)


if __name__ == '__main__':
    # 测试用例
    queries = [
        "Step 1: Analyze the clinical presentation...\n\nThe final answer is: $\\boxed{A}$",
        "After careful consideration...\n\nThe answer is \\boxed{B}.",
        "Based on the symptoms...\n\nThe final answer is: $\\boxed{B}$",
        "Clinical analysis shows...\n\nThe correct answer is A."
    ]
    
    labels = ["A", "A", "B", "A"]  # 在多数投票中不使用，但保持接口一致性
    
    rewards = reward_func(queries, [], labels)
    print(f"Extracted answers: {[extract_answer_from_response(q) for q in queries]}")
    print(f"Rewards: {rewards}")
    print(f"Majority answer gets reward, minority gets 0")