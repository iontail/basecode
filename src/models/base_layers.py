import torch
import torch.nn as nn
from src.models.base_models import BaseModel

class conv1dBLC(BaseModel, nn.Conv1d):
    """
    1D Convolution layer that automatically handles BLC (Batch, Length, Channel) format
    Converts BLC to BCL if needed before applying convolution
    """
    def __init__(self, *args, **kwargs):
        nn.Conv1d.__init__(self, *args, **kwargs)
        
    @property
    def dim(self) -> int:
        return self.out_channels
        
        
    def init_weights(self) -> None:
        """Initialize weights for conv1dBLC"""
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic format handling
        Args:
            - x: input tensor (B, C, L) or (B, L, C)
        Returns:
            - output: convolution output (B, out_channels, L_out)
        """
        if x.ndim != 3:
            raise ValueError(f'Expected 3D input (B, C, L) or (B, L, C), got {x.shape}')
        
        if x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2).contiguous()
        
        return super().forward(x) 


class conv2dBLC(BaseModel, nn.Conv2d):
    """
    2D Convolution layer that automatically handles BHWC format
    Converts BHWC to BCHW if needed before applying convolution
    """
    def __init__(self, *args, **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
        
    @property
    def dim(self) -> int:
        return self.out_channels
        
        
    def init_weights(self) -> None:
        """Initialize weights for conv2dBLC"""
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic format handling
        Args:
            - x: input tensor (B, C, H, W) or (B, H, W, C)
        Returns:
            - output: convolution output (B, out_channels, H_out, W_out)
        """
        if x.ndim != 4:
            raise ValueError(f'Expected 4D input (B, C, H, W) or (B, H, W, C), got {x.shape}')
        
        if x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        return super().forward(x)

    
class linearBLC(BaseModel, nn.Linear):
    """
    Linear layer that automatically handles feature dimension placement
    Supports features in last dimension or second dimension
    """
    def __init__(self, *args, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        
    @property
    def dim(self) -> int:
        return self.out_features
        
        
    def init_weights(self) -> None:
        """Initialize weights for linearBLC"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic feature dimension handling
        Args:
            - x: input tensor with features in last or second dimension
        Returns:
            - output: linear transformation output
        """
        if x.ndim < 2:
            raise ValueError(f"Expected input to have at least 2 dimensions, got {x.shape}")

        if x.shape[-1] != self.in_features:
            if x.shape[1] == self.in_features:
                x = x.transpose(1, -1).contiguous()
            else:
                raise ValueError(f"Expected last or second dim to be in_features={self.in_features}, got {x.shape}")

        return super().forward(x)