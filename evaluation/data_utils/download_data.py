#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face Dataset Downloader

这个脚本用于下载Hugging Face数据集到本地目录，支持多种格式和配置选项。

使用示例:
    python download_data.py --dataset_name "squad" --output_dir "./data" --split "train"
    python download_data.py --dataset_name "glue" --dataset_config "cola" --output_dir "./data"
    python download_data.py --dataset_name "wmt16" --dataset_config "de-en" --split "train" --format "json"
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Union

try:
    from datasets import load_dataset, Dataset, DatasetDict
    from huggingface_hub import snapshot_download, login
except ImportError as e:
    print(f"Error: Missing required packages. Please install with:")
    print(f"pip install datasets huggingface_hub")
    raise e

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HuggingFaceDatasetDownloader:
    """Hugging Face数据集下载器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化下载器
        
        Args:
            cache_dir: 缓存目录路径，如果为None则使用默认缓存目录
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ['HF_DATASETS_CACHE'] = cache_dir
    
    def authenticate(self, token: Optional[str] = None):
        """
        认证Hugging Face账户（用于私有数据集）
        
        Args:
            token: Hugging Face访问令牌
        """
        try:
            if token:
                login(token=token)
            else:
                login()  # 使用已保存的令牌或交互式登录
            logger.info("Successfully authenticated with Hugging Face")
        except Exception as e:
            logger.warning(f"Authentication failed: {e}")
    
    def download_dataset(
        self,
        dataset_name: str,
        output_dir: str,
        dataset_config: Optional[str] = None,
        split: Optional[Union[str, List[str]]] = None,
        streaming: bool = False,
        trust_remote_code: bool = False,
        **kwargs
    ) -> Union[Dataset, DatasetDict]:
        """
        下载Hugging Face数据集
        
        Args:
            dataset_name: 数据集名称（如 'squad', 'glue', 'wmt16'）
            output_dir: 输出目录
            dataset_config: 数据集配置名称（如 'cola' for glue）
            split: 数据集分割（如 'train', 'validation', 'test'）
            streaming: 是否使用流式加载
            trust_remote_code: 是否信任远程代码
            **kwargs: 其他传递给load_dataset的参数
        
        Returns:
            加载的数据集对象
        """
        try:
            logger.info(f"开始下载数据集: {dataset_name}")
            if dataset_config:
                logger.info(f"使用配置: {dataset_config}")
            if split:
                logger.info(f"下载分割: {split}")
            
            # 加载数据集
            dataset = load_dataset(
                dataset_name,
                name=dataset_config,
                split=split,
                streaming=streaming,
                trust_remote_code=trust_remote_code,
                cache_dir=self.cache_dir,
                **kwargs
            )
            
            logger.info(f"数据集下载完成: {dataset_name}")
            
            # 如果不是流式加载，保存到本地
            if not streaming:
                self._save_dataset_info(dataset, output_dir, dataset_name, dataset_config, split)
            
            return dataset
            
        except Exception as e:
            logger.error(f"下载数据集失败: {e}")
            raise
    
    def save_to_format(
        self,
        dataset: Union[Dataset, DatasetDict],
        output_dir: str,
        format_type: str = "json",
        filename_prefix: str = "dataset"
    ):
        """
        将数据集保存为指定格式
        
        Args:
            dataset: 数据集对象
            output_dir: 输出目录
            format_type: 输出格式 ('json', 'csv', 'parquet', 'arrow')
            filename_prefix: 文件名前缀
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(dataset, DatasetDict):
                # 处理多个分割的数据集
                for split_name, split_dataset in dataset.items():
                    filename = f"{filename_prefix}_{split_name}.{format_type}"
                    filepath = output_path / filename
                    self._save_single_dataset(split_dataset, filepath, format_type)
                    logger.info(f"保存 {split_name} 分割到: {filepath}")
            else:
                # 处理单个数据集
                filename = f"{filename_prefix}.{format_type}"
                filepath = output_path / filename
                self._save_single_dataset(dataset, filepath, format_type)
                logger.info(f"保存数据集到: {filepath}")
                
        except Exception as e:
            logger.error(f"保存数据集失败: {e}")
            raise
    
    def _save_single_dataset(self, dataset: Dataset, filepath: Path, format_type: str):
        """保存单个数据集到指定格式"""
        if format_type.lower() == "json":
            dataset.to_json(str(filepath))
        elif format_type.lower() == "csv":
            dataset.to_csv(str(filepath))
        elif format_type.lower() == "parquet":
            dataset.to_parquet(str(filepath))
        elif format_type.lower() == "arrow":
            dataset.save_to_disk(str(filepath.with_suffix("")))
        else:
            raise ValueError(f"不支持的格式: {format_type}")
    
    def _save_dataset_info(self, dataset, output_dir: str, dataset_name: str, 
                          dataset_config: Optional[str], split: Optional[str]):
        """保存数据集信息"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        info_file = output_path / "dataset_info.txt"
        
        with open(info_file, "w", encoding="utf-8") as f:
            f.write(f"Dataset Name: {dataset_name}\n")
            if dataset_config:
                f.write(f"Dataset Config: {dataset_config}\n")
            if split:
                f.write(f"Split: {split}\n")
            
            if isinstance(dataset, DatasetDict):
                f.write(f"\nDataset Splits:\n")
                for split_name, split_dataset in dataset.items():
                    f.write(f"  {split_name}: {len(split_dataset)} examples\n")
                    f.write(f"  Features: {list(split_dataset.features.keys())}\n")
            else:
                f.write(f"\nDataset Size: {len(dataset)} examples\n")
                f.write(f"Features: {list(dataset.features.keys())}\n")
    
    def download_raw_files(
        self,
        repo_id: str,
        output_dir: str,
        repo_type: str = "dataset",
        revision: str = "main",
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ):
        """
        下载数据集仓库的原始文件
        
        Args:
            repo_id: 仓库ID（如 'squad', 'glue'）
            output_dir: 输出目录
            repo_type: 仓库类型 ('dataset', 'model', 'space')
            revision: 版本/分支名称
            allow_patterns: 允许下载的文件模式
            ignore_patterns: 忽略的文件模式
        """
        try:
            logger.info(f"开始下载仓库文件: {repo_id}")
            
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_dir=output_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns
            )
            
            logger.info(f"仓库文件下载完成: {output_dir}")
            
        except Exception as e:
            logger.error(f"下载仓库文件失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载Hugging Face数据集到本地目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --dataset_name "squad" --output_dir "./data"
  %(prog)s --dataset_name "glue" --dataset_config "cola" --output_dir "./data"
  %(prog)s --dataset_name "wmt16" --dataset_config "de-en" --split "train" --format "json"
  %(prog)s --repo_id "squad" --output_dir "./raw_data" --download_raw
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="数据集名称（如 'squad', 'glue', 'wmt16'）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--dataset_config",
        type=str,
        help="数据集配置名称（如 'cola' for glue）"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        help="数据集分割（如 'train', 'validation', 'test'），多个分割用逗号分隔"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv", "parquet", "arrow"],
        help="输出格式（默认: json）"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="缓存目录路径"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="使用流式加载（适用于大型数据集）"
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="信任远程代码"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face访问令牌（用于私有数据集）"
    )
    
    # 原始文件下载选项
    parser.add_argument(
        "--download_raw",
        action="store_true",
        help="下载仓库的原始文件而不是处理后的数据集"
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        help="仓库ID（用于原始文件下载）"
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="版本/分支名称（默认: main）"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.download_raw:
        if not args.repo_id:
            parser.error("--download_raw 需要指定 --repo_id")
    else:
        if not args.dataset_name:
            parser.error("需要指定 --dataset_name 或使用 --download_raw")
    
    try:
        # 创建下载器
        downloader = HuggingFaceDatasetDownloader(cache_dir=args.cache_dir)
        
        # 认证（如果提供了token）
        if args.token:
            downloader.authenticate(args.token)
        
        if args.download_raw:
            # 下载原始文件
            downloader.download_raw_files(
                repo_id=args.repo_id,
                output_dir=args.output_dir,
                revision=args.revision
            )
        else:
            # 处理split参数
            split = None
            if args.split:
                split_list = [s.strip() for s in args.split.split(",")]
                split = split_list if len(split_list) > 1 else split_list[0]
            
            # 下载数据集
            dataset = downloader.download_dataset(
                dataset_name=args.dataset_name,
                output_dir=args.output_dir,
                dataset_config=args.dataset_config,
                split=split,
                streaming=args.streaming,
                trust_remote_code=args.trust_remote_code
            )
            
            # 保存为指定格式（如果不是流式加载）
            if not args.streaming:
                filename_prefix = args.dataset_name
                if args.dataset_config:
                    filename_prefix += f"_{args.dataset_config}"
                
                downloader.save_to_format(
                    dataset=dataset,
                    output_dir=args.output_dir,
                    format_type=args.format,
                    filename_prefix=filename_prefix
                )
        
        logger.info("所有操作完成！")
        
    except Exception as e:
        logger.error(f"操作失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())