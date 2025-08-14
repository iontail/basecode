from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, ExponentialLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
import logging



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
        
        # Mixed precision training
        self.scaler = GradScaler() if args.mixed_precision else None
        
        # Early stopping
        self.early_stopping_counter = 0
        
        # Logging
        self.writer = None
        self._setup_logging()
        
        self._setup_reproducibility()
        self._setup_directories()
        
        # Multi-GPU support
        self._setup_multigpu()
    
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
    
    def _setup_directories(self):
        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(self.args.log_dir, exist_ok=True)
    
    def _setup_logging(self):
        if self.args.use_tensorboard:
            log_path = os.path.join(self.args.log_dir, self.args.experiment_name)
            self.writer = SummaryWriter(log_path)
    
    def _setup_multigpu(self):
        if self.args.use_multigpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        if self.args.compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
    
    
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
        elif self.args.optimizer == 'RMSprop':
            momentum = getattr(self.args, 'momentum', 0.0)
            optimizer = RMSprop(
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
            eta_min = getattr(self.args, 'min_lr', 0)
            return CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=eta_min)
        elif self.args.scheduler == 'step':
            step_size = getattr(self.args, 'step_size', self.args.epochs // 3)
            gamma = getattr(self.args, 'gamma', 0.1)
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.args.scheduler == 'plateau':
            patience = getattr(self.args, 'scheduler_patience', 10)
            factor = getattr(self.args, 'scheduler_factor', 0.1)
            return ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
        elif self.args.scheduler == 'exponential':
            gamma = getattr(self.args, 'gamma', 0.95)
            return ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unsupported scheduler: {self.args.scheduler}")
    
    
    def save_checkpoint(self, filepath: str, is_best: bool = False, additional_info: Dict[str, Any] = None):
        # Handle DataParallel models
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'args': self.args,
            'early_stopping_counter': self.early_stopping_counter
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Handle DataParallel models
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', None)
        self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        
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
            
            # Log to TensorBoard
            if self.writer:
                for metric_name, value in phase_metrics.items():
                    self.writer.add_scalar(f"{phase}/{metric_name}", value, epoch)
        
        if phase_parts:
            print(f"Epoch {epoch} | {' | '.join(phase_parts)}")
        
        # Log learning rate
        if self.writer and self.optimizer:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('learning_rate', current_lr, epoch)
    
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
                    self.early_stopping_counter = 0
                    self.save_checkpoint(
                        os.path.join(self.args.save_dir, f'best_epoch_{epoch}.pt'), 
                        is_best=True
                    )
                elif self.args.early_stopping:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.args.patience:
                        print(f"Early stopping triggered after {self.args.patience} epochs without improvement")
                        break
            
            # Save regular checkpoint
            if hasattr(self.args, 'save_freq') and (epoch + 1) % self.args.save_freq == 0:
                checkpoint_path = os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch}.pt')
                self.save_checkpoint(checkpoint_path)
        
        if self.writer:
            self.writer.close()
        
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
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints if exceeding keep_checkpoint_max"""
        if not hasattr(self.args, 'keep_checkpoint_max') or self.args.keep_checkpoint_max <= 0:
            return
        
        checkpoint_files = []
        for file in os.listdir(self.args.save_dir):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
                filepath = os.path.join(self.args.save_dir, file)
                checkpoint_files.append((filepath, os.path.getmtime(filepath)))
        
        if len(checkpoint_files) > self.args.keep_checkpoint_max:
            checkpoint_files.sort(key=lambda x: x[1])
            for filepath, _ in checkpoint_files[:-self.args.keep_checkpoint_max]:
                os.remove(filepath)
                print(f"Removed old checkpoint: {os.path.basename(filepath)}")
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        if self.optimizer:
            return self.optimizer.param_groups[0]['lr']
        return 0.0
    
    def clip_gradients(self):
        """Clip gradients if grad_clip is specified"""
        if hasattr(self.args, 'grad_clip') and self.args.grad_clip > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
    
    def mixed_precision_step(self, loss: torch.Tensor):
        """Perform optimization step with mixed precision support"""
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.clip_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.clip_gradients()
            self.optimizer.step()
        
        self.optimizer.zero_grad()