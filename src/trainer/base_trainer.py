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
            optimizer = SGD(
                params,
                lr=self.args.lr,
                momentum=self.args.momentum,
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
            return StepLR(optimizer, step_size=self.args.epochs // 3, gamma=0.1)
        elif self.args.scheduler == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
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
    
    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.current_epoch
        
        metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        print(f"Epoch {epoch} | {metric_str}")
    
    def model_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def train(self):
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer)
        self.criterion = self.get_criterion()
        
        if hasattr(self.args, 'resume') and self.args.resume:
            self.load_checkpoint(self.args.resume)

        self.model_summary()
        
        print(f"Starting training for {self.args.epochs} epochs...")
        
        for epoch in range(self.current_epoch, self.args.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            eval_freq = getattr(self.args, 'eval_freq', 1)
            if epoch % eval_freq == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {}
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
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
        pass
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    @abstractmethod
    def inference(self, batch) -> torch.Tensor:
        pass