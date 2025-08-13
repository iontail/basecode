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
        for param in self.model.parameters():
            size += param.nelement() * param.element_size()
        for buf in self.model.buffers():
            size += buf.nelement() * buf.element_size()
        return size
    
    def model_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_bytes = self.model_size_b()
        model_size_mb = model_size_bytes / (1024 ** 2)
        
        print(f"Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size_mb:.2f} MiB ({model_size_bytes:,} bytes)")
    
