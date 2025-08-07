from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import os
import math
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, ExponentialLR, CosineAnnealingWarmRestarts

from contextlib import nullcontext
from tqdm.auto import tqdm
import random
import numpy as np
import wandb

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, metric: float) -> bool:
        if self.best_metric is None:
            self.best_metric = metric
        elif self._is_better(metric):
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_metric - self.min_delta
        else:
            return metric > self.best_metric + self.min_delta


class BaseTrainer(ABC):
    """Abstract base trainer class that integrates with arguments and config system."""
    
    def __init__(self, model: nn.Module, args):
        self.model = model
        self.args = args
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        
        # Tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.best_model_path = None
        
        # Setup training environment
        self._setup_reproducibility()
        self._setup_precision()
        self._setup_logging()
        self._setup_early_stopping()
        
        # Create save directories
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Setup device based on arguments."""
        if self.args.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(self.args.device)
        
        print(f"Using device: {device}")
        return device
    
    def _setup_reproducibility(self):
        """Setup random seeds for reproducibility."""
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            
            if self.args.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True
    
    def _setup_precision(self):
        """Setup mixed precision training if enabled."""
        if self.args.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast_context = torch.cuda.amp.autocast
        else:
            self.scaler = None
            self.autocast_context = nullcontext
    
    def _setup_logging(self):
        """Setup logging utilities (W&B, TensorBoard)."""
        self.wandb_logger = None
        self.tb_logger = None
        
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=self.args.experiment_name,
                config=vars(self.args)
            )
            self.wandb_logger = wandb
        
        if self.args.use_tensorboard and TENSORBOARD_AVAILABLE:
            log_path = os.path.join(self.args.log_dir, self.args.experiment_name)
            self.tb_logger = SummaryWriter(log_path)
    
    def _setup_early_stopping(self):
        """Setup early stopping if enabled."""
        if self.args.early_stopping:
            self.early_stopper = EarlyStopping(
                patience=self.args.patience,
                min_delta=self.args.min_delta,
                mode="min"  # Assuming we want to minimize validation loss
            )
        else:
            self.early_stopper = None
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on arguments."""
        params = self.model.parameters()
        
        if self.args.optimizer == 'Adam':
            optimizer = Adam(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'AdamW':
            optimizer = AdamW(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'SGD':
            optimizer = SGD(
                params,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'RMSprop':
            optimizer = RMSprop(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        
        return optimizer
    
    # 여기부분 config.py에서 불러와서 사용하도록 수정해야함
    def get_scheduler(self, optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on arguments."""
        if self.args.scheduler == 'none':
            return None
        elif self.args.scheduler == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.min_lr
            )
        elif self.args.scheduler == 'step':
            return StepLR(
                optimizer,
                step_size=self.args.epochs // 3,
                gamma=0.1
            )
        elif self.args.scheduler == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=10,
                factor=0.1
            )
        elif self.args.scheduler == 'exponential':
            return ExponentialLR(optimizer, gamma=0.95)
        else:
            raise ValueError(f"Unsupported scheduler: {self.args.scheduler}")
    
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'args': self.args
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', None)
        
        print(f"Loaded checkpoint from {filepath}")
        return checkpoint
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to various loggers."""
        if step is None:
            step = self.global_step
        
        # Console logging
        if step % self.args.print_freq == 0:
            metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            print(f"Step {step} | {metric_str}")
        
        # W&B logging
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)
        
        # TensorBoard logging
        if self.tb_logger:
            for name, value in metrics.items():
                self.tb_logger.add_scalar(name, value, step)
    
    def model_size_b(self) -> int:
        """
        Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
        Args:
        - model: self-explanatory
        Returns:
        - size: model size in bytes
        """
        size = 0
        for param in self.model.parameters():
            size += param.nelement() * param.element_size()
        for buf in self.model.buffers():
            size += buf.nelement() * buf.element_size()
        return size
    
    def model_summary(self):
        """Print model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_bytes = self.model_size_b()
        model_size_mb = model_size_bytes / (1024 ** 2) # MiB = 1024 * 1024 bytes
        
        print(f"Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size_mb:.2f} MiB ({model_size_bytes:,} bytes)")

    def cleanup(self):
        """Cleanup resources."""
        if self.wandb_logger:
            wandb.finish()
        if self.tb_logger:
            self.tb_logger.close()
    
    def train(self):
        """Main training loop."""
        # Initialize training components
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer)
        self.criterion = self.get_criterion()
        
        # Resume from checkpoint if specified
        if self.args.resume:
            self.load_checkpoint(self.args.resume)

        # Print model summary
        self.model_summary()
        
        
        
        # Compile model if requested (PyTorch 2.0+)
        if self.args.compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        print(f"Starting training for {self.args.epochs} epochs...")
        
        for epoch in range(self.current_epoch, self.args.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            if epoch % self.args.eval_freq == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {}
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            self.log_metrics(all_metrics)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if 'val_loss' in val_metrics:
                        self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Save checkpoints
            if self.args.save_last:
                last_path = os.path.join(self.args.save_dir, 'last_checkpoint.pt')
                self.save_checkpoint(last_path)
            
            # Save best model
            if self.args.save_best and 'val_loss' in val_metrics:
                is_best = (self.best_metric is None or 
                          val_metrics['val_loss'] < self.best_metric)
                if is_best:
                    self.best_metric = val_metrics['val_loss']
                    self.save_checkpoint(
                        os.path.join(self.args.save_dir, f'best_epoch_{epoch}.pt'), 
                        is_best=True
                    )
            
            # Early stopping
            if self.early_stopper and 'val_loss' in val_metrics:
                if self.early_stopper(val_metrics['val_loss']):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        print("Training completed!")
        self.cleanup()
    
    # Abstract methods that must be implemented by subclasses
        
    @abstractmethod
    def get_criterion(self) -> nn.Module:
        """Create loss criterion. Should be implemented by subclass if needed."""
        pass
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Must return dict of metrics."""
        pass
    
    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch. Must return dict of metrics."""
        pass
    
    @abstractmethod
    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Must return (loss, predictions)."""
        pass
    
    @abstractmethod
    def inference(self, batch) -> torch.Tensor:
        """Run inference on a batch. Should be implemented by subclass if needed."""
        pass