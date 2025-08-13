import torch
import torch.nn as nn
from base_trainer import BaseTrainer
from typing import Any, Dict, Optional, Tuple, Union


# Define your main trainer class inheriting from BaseTrainer
class MainTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, args, train_loader=None, val_loader=None, test_loader=None):
        super(MainTrainer, self).__init__(model, args, train_loader, val_loader, test_loader)

    def get_criterion(self) -> nn.Module:
        """Create loss criterion. Should be implemented by subclass if needed."""
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Must return dict of metrics."""
        pass
    
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch. Must return dict of metrics."""
        pass
    
    
    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Must return (loss, predictions)."""
        pass
    

    def inference(self, batch) -> torch.Tensor:
        """Run inference on a batch. Should be implemented by subclass if needed."""
        pass


