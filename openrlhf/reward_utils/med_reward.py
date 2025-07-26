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

def reward_func(queries: list[str], prompts: list[str], labels: list[str])->torch.Tensor:
    reward_list = []
    for query, label in zip(queries, labels):
        correct_answer = label.upper()
        predicted_answer = extract_answer_from_response(query)
        
        # 判断是否正确
        is_correct = False
        if predicted_answer and predicted_answer == correct_answer:
            is_correct = True
        
        reward = 1.0 if is_correct else 0.0
        reward_list.append(reward)

    return torch.tensor(reward_list)


if __name__ == '__main__':
    queries = [
        ".\n\nStep 1: Identify the key clinical presentation of the patient, which in this case is the appearance of her nails.\n\nStep 2: Recognize that the patient's nail appearance is likely related to a systemic condition, as indicated by the physician's comment about the patient's embarrassment about the appearance of her nails.\n\nStep 3: Analyze the options and determine which one is most likely to be associated with the patient's nail appearance. Options A, B, C, D, and E describe different dermatological and systemic conditions, so we need to consider which one is most closely related to nail appearance.\n\nStep 4: Recognize that option A, Silvery plaques on extensor surfaces, is commonly associated with psoriasis, which is a condition that can cause changes to the nails, such as pitting, thickening, or separation from the nail bed.\n\nStep 5: Eliminate options B, C, and D, which describe conditions that are not typically associated with nail changes. Option B, Flesh-colored papules in the lumbosacral region, is associated with lichen planus, option C, Erosions of the dental enamel, is associated with systemic lupus erythematosus, and option D, Pallor of the conjunctival mucosa, is associated with anemia.\n\nStep 6: Recognize that option E, Holosystolic murmur at the left lower sternal border, is associated with mitral valve prolapse, which is not typically associated with nail changes.\n\nStep 7: Since the patient's nail appearance is most likely related to a systemic condition, option A is the most likely correct answer.\n\nThe final answer is: $\\boxed{A}$"
    ]
    labels = [
        "C"
    ]
    print(reward_func(queries, [], labels))
    