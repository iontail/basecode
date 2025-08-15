import torch
import torch.nn as nn

class conv1dBLC(nn.Conv1d):
    """
    1D Convolution layer that automatically handles BLC (Batch, Length, Channel) format
    Converts BLC to BCL if needed before applying convolution
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize conv1dBLC layer
        Args:
            - *args: arguments for nn.Conv1d
            - **kwargs: keyword arguments for nn.Conv1d
        """
        super().__init__(*args, **kwargs)
        
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


class conv2dBLC(nn.Conv2d):
    """
    2D Convolution layer that automatically handles BHWC format
    Converts BHWC to BCHW if needed before applying convolution
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize conv2dBLC layer
        Args:
            - *args: arguments for nn.Conv2d
            - **kwargs: keyword arguments for nn.Conv2d
        """
        super().__init__(*args, **kwargs)
        
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

    
class linearBLC(nn.Linear):
    """
    Linear layer that automatically handles feature dimension placement
    Supports features in last dimension or second dimension
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize linearBLC layer
        Args:
            - *args: arguments for nn.Linear
            - **kwargs: keyword arguments for nn.Linear
        """
        super().__init__(*args, **kwargs)

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