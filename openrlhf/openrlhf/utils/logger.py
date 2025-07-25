#!/usr/bin/env python3
"""
统一的日志记录工具，支持 WandB 和 SwanLab
"""

import os
from typing import Dict, Any, Optional


class UnifiedLogger:
    """
    统一的日志记录器，支持同时使用 WandB 和 SwanLab
    """
    
    def __init__(self, strategy_args, is_rank_0: bool = True):
        self.args = strategy_args
        self.is_rank_0 = is_rank_0
        self._wandb = None
        self._swanlab = None
        self._tensorboard = None
        # 初始化 WandB
        if hasattr(self.args, 'use_wandb') and self.args.use_wandb and self.is_rank_0:
            self._init_wandb()
            
        # 初始化 SwanLab
        if hasattr(self.args, 'use_swanlab') and self.args.use_swanlab and self.is_rank_0:
            self._init_swanlab()
            
        # 初始化 TensorBoard（当没有其他日志工具时）
        if (hasattr(self.args, 'use_tensorboard') and self.args.use_tensorboard and 
            self._wandb is None and self._swanlab is None and self.is_rank_0):
            self._init_tensorboard()
    
    def _init_wandb(self):
        """初始化 WandB"""
        try:
            import wandb
            
            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.args.use_wandb)
            
            wandb.init(
                entity=getattr(self.args, 'wandb_org', None),
                project=getattr(self.args, 'wandb_project', 'openrlhf_train'),
                group=getattr(self.args, 'wandb_group', None),
                name=getattr(self.args, 'wandb_run_name', None),
                config=self.args.__dict__,
            )
            
            # 定义指标
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
            
            print("✅ WandB initialized successfully")
            
        except ImportError:
            print("❌ WandB not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"❌ Failed to initialize WandB: {e}")
    
    def _init_swanlab(self):
        """初始化 SwanLab"""
        try:
            import swanlab
            
            self._swanlab = swanlab
            
            # SwanLab 初始化参数
            init_kwargs = {
                'project': getattr(self.args, 'swanlab_project', getattr(self.args, 'wandb_project', 'openrlhf_train')),
                'experiment_name': getattr(self.args, 'swanlab_run_name', getattr(self.args, 'wandb_run_name', None)),
                'config': self.args.__dict__,
            }
            
            # 添加可选参数
            if hasattr(self.args, 'swanlab_workspace'):
                init_kwargs['workspace'] = self.args.swanlab_workspace
            if hasattr(self.args, 'use_swanlab') and isinstance(self.args.use_swanlab, str):
                # 如果 use_swanlab 是 API key
                os.environ['SWANLAB_API_KEY'] = self.args.use_swanlab
            
            swanlab.init(**init_kwargs)
            
            print("✅ SwanLab initialized successfully")
            
        except ImportError:
            print("❌ SwanLab not installed. Install with: pip install swanlab")
        except Exception as e:
            print(f"❌ Failed to initialize SwanLab: {e}")
    
    def _init_tensorboard(self):
        """初始化 TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            os.makedirs(self.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(
                self.args.use_tensorboard, 
                getattr(self.args, 'wandb_run_name', 'default_run')
            )
            self._tensorboard = SummaryWriter(log_dir=log_dir)
            
            print(f"✅ TensorBoard initialized successfully, log_dir: {log_dir}")
            
        except ImportError:
            print("❌ TensorBoard not available")
        except Exception as e:
            print(f"❌ Failed to initialize TensorBoard: {e}")
    
    def log(self, logs: Dict[str, Any], step: Optional[int] = None):
        """记录日志到所有已初始化的日志工具"""
        if not self.is_rank_0:
            return
            
        # 记录到 WandB
        if self._wandb is not None:
            try:
                if step is not None:
                    logs_with_step = {**logs, 'global_step': step}
                    self._wandb.log(logs_with_step)
                else:
                    self._wandb.log(logs)
            except Exception as e:
                print(f"❌ Failed to log to WandB: {e}")
        
        # 记录到 SwanLab
        if self._swanlab is not None:
            try:
                if step is not None:
                    self._swanlab.log(logs, step=step)
                else:
                    self._swanlab.log(logs)
            except Exception as e:
                print(f"❌ Failed to log to SwanLab: {e}")
        
        # 记录到 TensorBoard
        if self._tensorboard is not None and step is not None:
            try:
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        self._tensorboard.add_scalar(key, value, step)
            except Exception as e:
                print(f"❌ Failed to log to TensorBoard: {e}")
    
    def finish(self):
        """结束所有日志记录"""
        if not self.is_rank_0:
            return
            
        if self._wandb is not None:
            try:
                self._wandb.finish()
                print("✅ WandB finished")
            except Exception as e:
                print(f"❌ Failed to finish WandB: {e}")
        
        if self._swanlab is not None:
            try:
                self._swanlab.finish()
                print("✅ SwanLab finished")
            except Exception as e:
                print(f"❌ Failed to finish SwanLab: {e}")
        
        if self._tensorboard is not None:
            try:
                self._tensorboard.close()
                print("✅ TensorBoard closed")
            except Exception as e:
                print(f"❌ Failed to close TensorBoard: {e}")
    
    def is_available(self) -> bool:
        """检查是否有任何日志工具可用"""
        return any([self._wandb is not None, self._swanlab is not None, self._tensorboard is not None])
    
    def get_available_loggers(self) -> list:
        """获取可用的日志工具列表"""
        loggers = []
        if self._wandb is not None:
            loggers.append('wandb')
        if self._swanlab is not None:
            loggers.append('swanlab')
        if self._tensorboard is not None:
            loggers.append('tensorboard')
        return loggers