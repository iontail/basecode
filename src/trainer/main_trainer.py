import torch
import torch.nn as nn
from src.trainer.base_trainer import BaseTrainer
from typing import Any, Dict, Optional, Tuple, Union

# Import user-defined loss function
try:
    from src.utils.loss import get_loss_function
    CUSTOM_LOSS_AVAILABLE = True
except ImportError:
    CUSTOM_LOSS_AVAILABLE = False
    print("Warning: src/utils/loss.py not found or get_loss_function() not defined. Using default loss.")


class MainTrainer(BaseTrainer):
    """
    Main trainer implementation inheriting from BaseTrainer
    Provides concrete implementations of abstract methods for specific training tasks
    """
    def __init__(self, model: nn.Module, args, train_loader=None, val_loader=None, test_loader=None):
        """
        Initialize MainTrainer
        Args:
            - model: PyTorch model to train
            - args: training arguments/configuration
            - train_loader: training data loader (optional)
            - val_loader: validation data loader (optional)
            - test_loader: test data loader (optional)
        """
        
        super(MainTrainer, self).__init__(model, args, train_loader, val_loader, test_loader)
        
        """
        To apply different learning rates to specific layers, override self.params after super().__init__()
        - Example for different LRs:
        self.params = [
            {'params': self.model.backbone.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': self.model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 1e-3}
        ]

        - Example for layer-wise LR decay (BERT-style):
        self.params = []
        for i, layer in enumerate(self.model.layers):
            lr = self.args.lr * (self.args.layer_lr_decay ** (len(self.model.layers) - i))
            self.params.append({'params': layer.parameters(), 'lr': lr})
        """

    def get_criterion(self):
        """
        Return the loss criteria for training
        Gets loss function from loss.py if available, otherwise uses default
        Returns:
            - Loss function(s) defined in loss.py or default CrossEntropyLoss
        """
        if CUSTOM_LOSS_AVAILABLE:
            try:
                return get_loss_function(self.args)
            except Exception as e:
                print(f"Error loading custom loss function: {e}")
                print("Falling back to default CrossEntropyLoss")
        
        # Default fallback loss
        label_smoothing = getattr(self.args, 'label_smoothing', 0.0)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Execute one training epoch and return training metrics
        Args:
            - train_loader: training data loader
        Returns:
            - metrics: dictionary of training metrics (e.g., {'loss': loss_value})
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            loss, _ = self.forward_pass(batch)
            
            self.mixed_precision_step(loss)
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """
        Execute validation and return validation metrics
        Args:
            - val_loader: validation data loader
        Returns:
            - metrics: dictionary of validation metrics (e.g., {'val_loss': loss_value})
        """
        if val_loader is None:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = self.forward_pass(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return {'val_loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through model and compute loss
        Args:
            - batch: batch of data from data loader
        Returns:
            - tuple: (loss, predictions)
        """
        # Move batch to device
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # Forward pass
        predictions = self.model(images)
        
        # Compute loss
        loss = self.criteria(predictions, labels)
        
        return loss, predictions
    
    def inference(self, batch) -> torch.Tensor:
        """
        Run inference without gradient computation
        Args:
            - batch: batch of data from data loader
        Returns:
            - predictions: model predictions
        """
        self.model.eval()
        with torch.no_grad():
            images = batch['image'].to(self.device)
            predictions = self.model(images)
        return predictions