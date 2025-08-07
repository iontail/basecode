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
    
