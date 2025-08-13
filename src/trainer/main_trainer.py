import torch
import torch.nn as nn
from base_trainer import BaseTrainer
from typing import Any, Dict, Optional, Tuple, Union


class MainTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, args, train_loader=None, val_loader=None, test_loader=None):
        super(MainTrainer, self).__init__(model, args, train_loader, val_loader, test_loader)

    def get_criterion(self) -> nn.Module:
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        pass
    
    def validate_epoch(self) -> Dict[str, float]:
        pass
    
    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    def inference(self, batch) -> torch.Tensor:
        pass