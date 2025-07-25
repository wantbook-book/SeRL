#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上传本地模型到Hugging Face Hub的工具脚本

使用方法:
1. 安装依赖: pip install huggingface_hub transformers
2. 登录HF: huggingface-cli login
3. 运行脚本: python upload_huggingface.py
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_files(local_path):
    """
    检查本地模型目录是否包含必要的文件
    
    Args:
        local_path (str): 本地模型目录路径
        
    Returns:
        bool: 是否包含必要文件
    """
    required_files = ['config.json']
    optional_files = ['pytorch_model.bin', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']
    
    path = Path(local_path)
    if not path.exists():
        logger.error(f"本地路径不存在: {local_path}")
        return False
    
    # 检查必需文件
    for file in required_files:
        if not (path / file).exists():
            logger.error(f"缺少必需文件: {file}")
            return False
    
    # 检查可选文件
    has_model_file = any((path / file).exists() for file in optional_files)
    if not has_model_file:
        logger.warning("未找到模型权重文件，请确认是否正确")
    
    logger.info(f"模型目录检查通过: {local_path}")
    return True

def upload_model_to_hf(local_path, repo_name, token=None, private=False, commit_message=None):
    """
    上传本地模型到Hugging Face Hub
    
    Args:
        local_path (str): 本地模型目录路径
        repo_name (str): HF仓库名称 (格式: username/model-name)
        token (str, optional): HF访问令牌
        private (bool): 是否创建私有仓库
        commit_message (str, optional): 提交信息
    """
    try:
        # 初始化API
        api = HfApi(token=token)
        
        # 检查本地模型文件
        if not check_model_files(local_path):
            return False
        
        # 尝试创建仓库（如果不存在）
        try:
            logger.info(f"尝试创建仓库: {repo_name}")
            create_repo(
                repo_id=repo_name,
                token=token,
                private=private,
                exist_ok=True
            )
            logger.info(f"仓库创建成功或已存在: {repo_name}")
        except Exception as e:
            logger.error(f"创建仓库失败: {e}")
            return False
        
        # 上传模型文件
        logger.info(f"开始上传模型文件到 {repo_name}...")
        
        # 获取本地目录中的所有文件
        local_path = Path(local_path)
        files_to_upload = []
        
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                files_to_upload.append((str(file_path), str(relative_path)))
        
        logger.info(f"找到 {len(files_to_upload)} 个文件需要上传")
        
        # 批量上传文件
        for local_file, repo_file in files_to_upload:
            try:
                logger.info(f"上传文件: {repo_file}")
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=repo_file,
                    repo_id=repo_name,
                    token=token,
                    commit_message=commit_message or f"Upload {repo_file}"
                )
            except Exception as e:
                logger.error(f"上传文件 {repo_file} 失败: {e}")
                continue
        
        logger.info(f"模型上传完成! 🎉")
        logger.info(f"模型链接: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"上传过程中发生错误: {e}")
        return False

def upload_folder_to_hf(local_path, repo_name, token=None, private=False, commit_message=None):
    """
    使用folder上传方式（更高效）
    
    Args:
        local_path (str): 本地模型目录路径
        repo_name (str): HF仓库名称
        token (str, optional): HF访问令牌
        private (bool): 是否创建私有仓库
        commit_message (str, optional): 提交信息
    """
    try:
        api = HfApi(token=token)
        
        # 检查本地模型文件
        if not check_model_files(local_path):
            return False
        
        # 创建仓库
        try:
            create_repo(
                repo_id=repo_name,
                token=token,
                private=private,
                exist_ok=True
            )
            logger.info(f"仓库准备就绪: {repo_name}")
        except Exception as e:
            logger.error(f"创建仓库失败: {e}")
            return False
        
        # 上传整个文件夹
        logger.info(f"开始上传文件夹 {local_path} 到 {repo_name}...")
        
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            token=token,
            commit_message=commit_message or "Upload model files"
        )
        
        logger.info(f"文件夹上传完成! 🎉")
        logger.info(f"模型链接: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"上传过程中发生错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="上传本地模型到Hugging Face Hub")
    parser.add_argument("--local_path", type=str, required=True, help="本地模型目录路径")
    parser.add_argument("--repo_name", type=str, required=True, help="HF仓库名称 (格式: username/model-name)")
    parser.add_argument("--token", type=str, help="HF访问令牌 (可选，建议使用 huggingface-cli login)")
    parser.add_argument("--private", action="store_true", help="创建私有仓库")
    parser.add_argument("--commit_message", type=str, help="提交信息")
    parser.add_argument("--method", type=str, choices=["file", "folder"], default="folder", 
                       help="上传方式: file(逐个文件) 或 folder(整个文件夹)")
    
    args = parser.parse_args()
    
    # 验证参数
    if not os.path.exists(args.local_path):
        logger.error(f"本地路径不存在: {args.local_path}")
        return
    
    if "/" not in args.repo_name:
        logger.error("仓库名称格式错误，应为: username/model-name")
        return
    
    # 选择上传方式
    if args.method == "folder":
        success = upload_folder_to_hf(
            local_path=args.local_path,
            repo_name=args.repo_name,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message
        )
    else:
        success = upload_model_to_hf(
            local_path=args.local_path,
            repo_name=args.repo_name,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message
        )
    
    if success:
        logger.info("上传成功! ✅")
    else:
        logger.error("上传失败! ❌")

if __name__ == "__main__":
    # 示例用法
    print("="*50)
    print("Hugging Face 模型上传工具")
    print("="*50)
    print()
    print("使用示例:")
    print("python upload_huggingface.py --local_path /path/to/model --repo_name username/model-name")
    print("python upload_huggingface.py --local_path /path/to/model --repo_name username/model-name --private")
    print()
    print("注意事项:")
    print("1. 请先运行 'huggingface-cli login' 进行身份验证")
    print("2. 确保本地目录包含 config.json 等必要文件")
    print("3. 仓库名称格式: username/model-name")
    print()
    
    main()