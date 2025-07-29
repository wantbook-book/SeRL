#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward Model Loading and Inference

This module provides functionality to load the Qwen2.5-Math-1.5B-GRPO-RM reward model
and compute rewards for given responses.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardModel:
    """
    Reward Model wrapper for Qwen2.5-Math-1.5B-GRPO-RM
    """
    
    def __init__(self, model_name: str = "Yen729/Qwen2.5-Math-1.5B-GRPO-RM", device: str = "auto"):
        """
        初始化Reward Model
        
        Args:
            model_name (str): HuggingFace模型名称
            device (str): 设备类型，"auto"、"cuda"或"cpu"
        """
        self.model_name = model_name
        
        # 自动选择设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading reward model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # 加载tokenizer和model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"  # 对于reward model通常使用left padding
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Reward model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load reward model: {e}")
            raise
    
    def get_reward(self, response: Union[str, List[str]], max_length: int = 2048) -> Union[float, List[float]]:
        """
        计算response的reward值
        
        Args:
            response (Union[str, List[str]]): 单个response字符串或response列表
            max_length (int): 最大序列长度
            
        Returns:
            Union[float, List[float]]: 单个reward值或reward值列表
        """
        # 处理单个字符串输入
        if isinstance(response, str):
            responses = [response]
            return_single = True
        else:
            responses = response
            return_single = False
            
        try:
            with torch.no_grad():
                # Tokenize输入
                inputs = self.tokenizer(
                    responses,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                
                # 移动到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 获取logits并计算reward
                logits = outputs.logits
                
                # 如果模型输出多个类别，通常取第一个维度作为reward
                if logits.shape[-1] == 1:
                    rewards = logits.squeeze(-1)
                else:
                    # 对于分类模型，可能需要特殊处理
                    rewards = logits[:, 0]  # 或者使用softmax等
                
                # 转换为CPU并获取数值
                rewards = rewards.cpu().float().tolist()
                
                # 返回结果
                if return_single:
                    return rewards[0]
                else:
                    return rewards
                    
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            raise
    
    def batch_get_reward(self, responses: List[str], batch_size: int = 8, max_length: int = 2048) -> List[float]:
        """
        批量计算reward值
        
        Args:
            responses (List[str]): response列表
            batch_size (int): 批处理大小
            max_length (int): 最大序列长度
            
        Returns:
            List[float]: reward值列表
        """
        all_rewards = []
        
        for i in range(0, len(responses), batch_size):
            batch_responses = responses[i:i + batch_size]
            batch_rewards = self.get_reward(batch_responses, max_length)
            all_rewards.extend(batch_rewards)
            
        return all_rewards


def load_reward_model(model_name: str = "Yen729/Qwen2.5-Math-1.5B-GRPO-RM", device: str = "auto") -> RewardModel:
    """
    便捷函数：加载reward model
    
    Args:
        model_name (str): HuggingFace模型名称
        device (str): 设备类型
        
    Returns:
        RewardModel: 初始化的reward model实例
    """
    return RewardModel(model_name, device)


def get_response_reward(response: str, model: RewardModel = None, **kwargs) -> float:
    """
    便捷函数：获取单个response的reward
    
    Args:
        response (str): 输入的response
        model (RewardModel, optional): 预加载的模型实例
        **kwargs: 传递给RewardModel的其他参数
        
    Returns:
        float: reward值
    """
    if model is None:
        model = load_reward_model(**kwargs)
    
    return model.get_reward(response)


if __name__ == "__main__":
    # 示例用法
    print("Loading Qwen2.5-Math-1.5B-GRPO-RM reward model...")
    
    # 初始化模型
    rm = load_reward_model(model_name="/pubshare/LLM/Yen729/Qwen2.5-Math-1.5B-GRPO-RM")
    
    # 测试单个response
    test_response = "To solve this problem, I need to find the value of x.\n\nGiven equation: 2x + 5 = 13\n\nSubtracting 5 from both sides: 2x = 8\n\nDividing by 2: x = 4\n\nTherefore, x = 4."
    
    reward = rm.get_reward(test_response)
    print(f"Response reward: {reward:.4f}")
    
    # 测试多个responses
    test_responses = [
        "The answer is 42.",
        "Let me solve this step by step. First, I'll identify the key information...",
        "I don't know the answer."
    ]
    
    rewards = rm.get_reward(test_responses)
    print("\nBatch rewards:")
    for i, (resp, rew) in enumerate(zip(test_responses, rewards)):
        print(f"Response {i+1}: {rew:.4f}")
        print(f"Text: {resp[:50]}...\n")