import torch
import torch.nn as nn
from base_trainer import BaseTrainer
from typing import Any, Dict, Optional, Tuple, Union


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

    def get_criterion(self):
        """
        Return the loss criteria for training
        Returns:
            - Single loss function for classification tasks
        """
        return nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Execute one training epoch and return training metrics
        Args:
            - train_loader: training data loader
        Returns:
            - metrics: dictionary of training metrics (e.g., {'loss': loss_value})
        """
        pass
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """
        Execute validation and return validation metrics
        Args:
            - val_loader: validation data loader
        Returns:
            - metrics: dictionary of validation metrics (e.g., {'val_loss': loss_value})
        """
        pass
    
    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through model and compute loss
        Args:
            - batch: batch of data from data loader
        Returns:
            - tuple: (loss, predictions)
        """
        pass
    
    def inference(self, batch) -> torch.Tensor:
        """
        Run inference without gradient computation
        Args:
            - batch: batch of data from data loader
        Returns:
            - predictions: model predictions
        """
        pass