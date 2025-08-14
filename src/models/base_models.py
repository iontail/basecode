import torch
import torch.nn as nn
import numpy as np

from typing import Any, Dict, Optional, Tuple, Union, List
from abc import abstractmethod, ABC

class BaseModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    @abstractmethod
    @property
    def dim(self) -> int:
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def freeze(self, layer_names: Optional[List[str]] = None) -> 'BaseModel':
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
        return self

    def unfreeze(self, layer_names: Optional[List[str]] = None) -> 'BaseModel':
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
        return self
    
    def model_size_b(self) -> int:
        size = 0
        for param in self.parameters():
            size += param.nelement() * param.element_size()
        for buf in self.buffers():
            size += buf.nelement() * buf.element_size()
        return size
    
    def model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_bytes = self.model_size_b()
        model_size_mb = model_size_bytes / (1024 ** 2)
        
        print(f"Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size_mb:.2f} MiB ({model_size_bytes:,} bytes)")
    
    def count_parameters(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def get_device(self) -> torch.device:
        """Get the device of the model"""
        return next(self.parameters()).device
    
    def to_device(self, device: Union[torch.device, str]) -> 'BaseModel':
        """Move model to device"""
        return self.to(device)
    
    def freeze_backbone(self, unfreeze_layers: Optional[List[str]] = None) -> 'BaseModel':
        """Freeze all parameters except specified layers"""
        for name, param in self.named_parameters():
            if unfreeze_layers and any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
        return self
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (useful for optimizer)"""
        return [p for p in self.parameters() if p.requires_grad]
    
