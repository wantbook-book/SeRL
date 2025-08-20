#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face模型下载工具
支持从Hugging Face Hub下载预训练模型和tokenizer
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, List

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("请先安装依赖: pip install transformers huggingface_hub")
    exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Hugging Face模型下载器"""
    
    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """
        初始化下载器
        
        Args:
            cache_dir: 模型缓存目录，默认为 ~/.cache/huggingfacea
            token: Hugging Face访问令牌，用于下载私有模型
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.token = token or os.getenv("HF_TOKEN")
        
        # 确保缓存目录存在
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"模型缓存目录: {self.cache_dir}")
    
    def download_model(self, 
                      model_name: str, 
                      local_dir: Optional[str] = None,
                      revision: str = "main",
                      ignore_patterns: Optional[List[str]] = None) -> str:
        """
        下载完整模型仓库
        
        Args:
            model_name: 模型名称，如 'microsoft/DialoGPT-medium'
            local_dir: 本地保存目录，如果为None则使用cache_dir
            revision: 模型版本/分支，默认为'main'
            ignore_patterns: 忽略的文件模式列表
            
        Returns:
            str: 模型本地路径
        """
        try:
            logger.info(f"开始下载模型: {model_name}")
            
            # 设置忽略模式，默认忽略一些大文件
            if ignore_patterns is None:
                ignore_patterns = [
                    "*.bin",  # 可选：忽略旧格式的权重文件
                    "*.h5",   # 忽略TensorFlow格式
                    "*.ot",   # 忽略ONNX格式
                    "*.msgpack",  # 忽略msgpack格式
                ]
            
            # 下载模型
            local_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.cache_dir,
                local_dir=local_dir,
                revision=revision,
                token=self.token,
                ignore_patterns=ignore_patterns
            )
            
            logger.info(f"模型下载完成: {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            raise
    
    def download_specific_file(self, 
                              model_name: str, 
                              filename: str,
                              local_dir: Optional[str] = None,
                              revision: str = "main") -> str:
        """
        下载模型仓库中的特定文件
        
        Args:
            model_name: 模型名称
            filename: 要下载的文件名
            local_dir: 本地保存目录
            revision: 模型版本/分支
            
        Returns:
            str: 文件本地路径
        """
        try:
            logger.info(f"下载文件: {model_name}/{filename}")
            
            file_path = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                cache_dir=self.cache_dir,
                local_dir=local_dir,
                revision=revision,
                token=self.token
            )
            
            logger.info(f"文件下载完成: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"下载文件失败: {e}")
            raise
    
    def load_model_and_tokenizer(self, model_name: str, local_path: Optional[str] = None):
        """
        加载模型和tokenizer
        
        Args:
            model_name: 模型名称
            local_path: 本地模型路径，如果为None则从缓存加载
            
        Returns:
            tuple: (model, tokenizer)
        """
        try:
            model_path = local_path or model_name
            logger.info(f"加载模型和tokenizer: {model_path}")
            
            # 加载配置
            config = AutoConfig.from_pretrained(model_path, token=self.token)
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                token=self.token,
                trust_remote_code=True
            )
            
            # 加载模型
            model = AutoModel.from_pretrained(
                model_path,
                token=self.token,
                trust_remote_code=True
            )
            
            logger.info("模型和tokenizer加载完成")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Hugging Face模型下载工具")
    parser.add_argument("model_name", help="模型名称，如 microsoft/DialoGPT-medium")
    parser.add_argument("--cache-dir", help="缓存目录路径")
    parser.add_argument("--local-dir", help="本地保存目录")
    parser.add_argument("--token", help="Hugging Face访问令牌")
    parser.add_argument("--revision", default="main", help="模型版本/分支")
    parser.add_argument("--file", help="下载特定文件")
    parser.add_argument("--load-test", action="store_true", help="下载后测试加载模型")
    parser.add_argument("--include-weights", action="store_true", help="包含模型权重文件")
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = ModelDownloader(cache_dir=args.cache_dir, token=args.token)
    
    try:
        if args.file:
            # 下载特定文件
            file_path = downloader.download_specific_file(
                model_name=args.model_name,
                filename=args.file,
                local_dir=args.local_dir,
                revision=args.revision
            )
            print(f"文件已下载到: {file_path}")
        else:
            # 下载完整模型
            ignore_patterns = None if args.include_weights else [
                "*.bin", "*.safetensors", "*.h5", "*.ot", "*.msgpack"
            ]
            
            model_path = downloader.download_model(
                model_name=args.model_name,
                local_dir=args.local_dir,
                revision=args.revision,
                ignore_patterns=ignore_patterns
            )
            print(f"模型已下载到: {model_path}")
            
            # 测试加载
            if args.load_test:
                logger.info("测试加载模型...")
                model, tokenizer = downloader.load_model_and_tokenizer(
                    args.model_name, model_path
                )
                print(f"模型类型: {type(model).__name__}")
                print(f"Tokenizer类型: {type(tokenizer).__name__}")
                print(f"词汇表大小: {tokenizer.vocab_size}")
                
    except Exception as e:
        logger.error(f"操作失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())