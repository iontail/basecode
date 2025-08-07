import torch
import torch.nn as nn
import numpy as np

from typing import Any, Dict, Optional, Tuple, Union
from abc import abstractmethod, ABC
from base_models import BaseModel

class MyModel(BaseModel):
    """
    My custom model implementation.
    """
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)

        self.hidden_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement the forward pass
        return x
    
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        # Implement the inference logic
        return x
    
    @property
    def dim(self) -> int:
        # Return the dimension of the model's output
        return self.hidden_dim