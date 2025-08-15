import torch
import torch.nn as nn
import numpy as np

from typing import Any, Dict, Optional, Tuple, Union, List
from abc import abstractmethod, ABC

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all model implementations
    Provides common functionality including parameter management, device handling,
    and model statistics
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize BaseModel
        Args:
            - *args: Variable length argument list
            - **kwargs: Arbitrary keyword arguments
        """
        super(BaseModel, self).__init__(*args, **kwargs)

    @abstractmethod
    @property
    def dim(self) -> int:
        """
        Get model output dimension
        Returns:
            - dim: output dimension of the model
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        Args:
            - x: input tensor
        Returns:
            - output: model output tensor
        """
        pass
    
    @abstractmethod
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference pass through the model without gradient computation
        Args:
            - x: input tensor
        Returns:
            - output: model inference output tensor
        """
        pass

    def freeze(self, layer_names: Optional[List[str]] = None) -> 'BaseModel':
        """
        Freeze model parameters to prevent gradient updates
        Args:
            - layer_names: list of layer names to freeze, if None freezes all parameters
        Returns:
            - self: BaseModel instance for method chaining
        """
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
        return self

    def unfreeze(self, layer_names: Optional[List[str]] = None) -> 'BaseModel':
        """
        Unfreeze model parameters to allow gradient updates
        Args:
            - layer_names: list of layer names to unfreeze, if None unfreezes all parameters
        Returns:
            - self: BaseModel instance for method chaining
        """
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
        return self
    
    def model_size_b(self) -> int:
        """
        Calculate total model size in bytes including parameters and buffers
        Returns:
            - size: total model size in bytes
        """
        size = 0
        for param in self.parameters():
            size += param.nelement() * param.element_size()
        for buf in self.buffers():
            size += buf.nelement() * buf.element_size()
        return size
    
    def model_summary(self):
        """
        Print comprehensive model summary including parameters and memory usage
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_bytes = self.model_size_b()
        model_size_mb = model_size_bytes / (1024 ** 2)
        
        print(f"Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size_mb:.2f} MiB ({model_size_bytes:,} bytes)")
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters
        Returns:
            - tuple: (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def get_device(self) -> torch.device:
        """
        Get the device where the model is currently located
        Returns:
            - device: torch device (cpu, cuda, mps, etc.)
        """
        return next(self.parameters()).device
    
    def to_device(self, device: Union[torch.device, str]) -> 'BaseModel':
        """
        Move model to specified device
        Args:
            - device: target device (torch.device or string)
        Returns:
            - self: BaseModel instance for method chaining
        """
        return self.to(device)
    
    def freeze_backbone(self, unfreeze_layers: Optional[List[str]] = None) -> 'BaseModel':
        """
        Freeze all parameters except specified layers
        Args:
            - unfreeze_layers: list of layer names to keep unfrozen
        Returns:
            - self: BaseModel instance for method chaining
        """
        for name, param in self.named_parameters():
            if unfreeze_layers and any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
        return self
    
    def get_trainable_parameters(self):
        """
        Get only trainable parameters (useful for optimizer initialization)
        Returns:
            - params: list of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
