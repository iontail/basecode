from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import os
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

import random
import numpy as np



class BaseTrainer(ABC):
    def __init__(self, model: nn.Module, args, train_loader=None, val_loader=None, test_loader=None):
        self.model = model
        self.args = args
        self.device = self._setup_device()
        self.model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.current_epoch = 0
        self.best_metric = None
        self.best_model_path = None
        
        self._setup_reproducibility()
        os.makedirs(args.save_dir, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
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
    
    
    def get_optimizer(self) -> torch.optim.Optimizer:
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
            momentum = getattr(self.args, 'momentum', 0.9)
            optimizer = SGD(
                params,
                lr=self.args.lr,
                momentum=momentum,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        
        return optimizer
    
    def get_scheduler(self, optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if not hasattr(self.args, 'scheduler') or self.args.scheduler == 'none':
            return None
        elif self.args.scheduler == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'step':
            step_size = getattr(self.args, 'step_size', self.args.epochs // 3)
            gamma = getattr(self.args, 'gamma', 0.1)
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.args.scheduler == 'plateau':
            patience = getattr(self.args, 'scheduler_patience', 10)
            factor = getattr(self.args, 'scheduler_factor', 0.1)
            return ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
        else:
            raise ValueError(f"Unsupported scheduler: {self.args.scheduler}")
    
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'args': self.args
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', None)
        
        print(f"Loaded checkpoint from {filepath}")
        return checkpoint
    
    def log_metrics(self, metrics: Dict[str, Dict[str, float]], epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.current_epoch

        phase_parts = []
        for phase, phase_metrics in metrics.items():
            if not phase_metrics:
                continue
            metric_items = [f"{k}: {v:.4f}" for k, v in phase_metrics.items()]
            phase_parts.append(f"{phase.upper()}: {' | '.join(metric_items)}")
        
        if phase_parts:
            print(f"Epoch {epoch} | {' | '.join(phase_parts)}")
    
    def model_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        MiB = 1024 ** 2
        size_bytes = self.model_size_b(self.model)
        size_mib = size_bytes / MiB
        
        print(f"Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {size_mib:.2f} MiB ({size_bytes:,} bytes)")
    
    def model_size_b(self, model: nn.Module) -> int:
        """
        Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
        Args:
        - model: self-explanatory
        Returns:
        - size: model size in bytes
        """
        size = 0
        for param in model.parameters():
            size += param.nelement() * param.element_size()
        for buf in model.buffers():
            size += buf.nelement() * buf.element_size()
        return size
    
    def train(self):
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer)
        self.criterion = self.get_criterion()
        
        if hasattr(self.args, 'resume') and self.args.resume:
            _ = self.load_checkpoint(self.args.resume, load_optimizer=True)

        self.model_summary()
        
        print(f"Starting training for {self.args.epochs} epochs...")

        eval_freq = getattr(self.args, 'eval_freq', 1)

        for epoch in range(self.current_epoch, self.args.epochs):
            self.current_epoch = epoch + 1
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            
            if epoch % eval_freq == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {}
            
            # Combine metrics
            all_metrics = {'train': train_metrics, 'val': val_metrics}
            
            # Log metrics
            self.log_metrics(all_metrics, epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if 'val_loss' in val_metrics:
                        self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Save checkpoints
            if hasattr(self.args, 'save_best') and self.args.save_best and 'val_loss' in val_metrics:
                is_best = (self.best_metric is None or val_metrics['val_loss'] < self.best_metric)
                if is_best:
                    self.best_metric = val_metrics['val_loss']
                    self.save_checkpoint(
                        os.path.join(self.args.save_dir, f'best_epoch_{epoch}.pt'), 
                        is_best=True
                    )
        
        print("Training completed!")
    
    @abstractmethod
    def get_criterion(self) -> nn.Module:
        """Return the loss criterion for training."""
        pass
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch and return training metrics."""
        pass
    
    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """Execute validation and return validation metrics."""
        pass
    
    @abstractmethod
    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through model and compute loss. Return (loss, predictions)."""
        pass
    
    @abstractmethod
    def inference(self, batch) -> torch.Tensor:
        """Run inference without gradient computation. Return predictions."""
        pass