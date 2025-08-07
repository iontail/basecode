import torch
import torch.nn as nn
import numpy as np

from typing import Any, Dict, Optional, Tuple, Union, List
from abc import abstractmethod, ABC

class BaseModel(nn.Module, ABC):
    """
    Base class for all models. Should be inherited by all model implementations.
    Provides basic structure and methods for model initialization and forward pass.
    """
    
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    @abstractmethod
    @property
    def dim(self) -> int:
        """
        Dimension of the model's output. Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference method for the model. Should be implemented by subclasses if needed.
        """
        pass

    # === Not Abstract Method ===
    def freeze(self, layer_names: Optional[List[str]] = None) -> 'BaseModel':
        """Freeze model parameters."""
        if layer_names is None:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
        return self


    def unfreeze(self, layer_names: Optional[List[str]] = None) -> 'BaseModel':
        """Unfreeze model parameters."""
        if layer_names is None:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
        return self
    
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
    
