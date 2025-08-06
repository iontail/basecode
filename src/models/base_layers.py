import torch
import torch.nn as nn

class conv1dBLC(nn.Conv1d):
    """
    Automatically transposes the input tensor if the 0th dimension is the number of channels.
    Args:
    - same as torch.nn.Conv1d
    Returns:
    - same as torch.nn.Conv1d 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the convolution

        if x.shape[1] != self.in_channels and x.shape[0] == self.in_channels:
            x = x.transpose(0, 1).contiguous()

        x = super().forward(x) 
        
        
        return x