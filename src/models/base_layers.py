import torch
import torch.nn as nn

class conv1dBLC(nn.Conv1d):
    """
    Automatically transposes input of shape (B, L, C) to (B, C, L) if needed.
    Accepts input of shape (B, C, L) or (B, L, C).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the convolution

        if x.ndim != 3:
            raise ValueError(f'Expected 3D input (B, C, L) or (B, L, C), got {x.shape}')
        
        if x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2).contiguous()
        
        return super().forward(x) 
    

class conv2dBLC(nn.Conv2d):
    """
    Automatically permutes input of shape (B, H, W, C) to (B, C, H, W) if needed.
    Accepts input of shape (B, C, H, W) or (B, H, W, C).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the convolution

        if x.ndim != 4:
            raise ValueError(f'Expected 4D input (B, C, H, W) or (B, H, W, C), got {x.shape}')
        
        if x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        return super().forward(x)
    
class linearBLC(nn.Linear):
    """
    Automatically transposes input of shape (B, D, ...) if needed to align last dim with in_features.
    Accepts input of shape (B, in_features), or (..., in_features).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected input to have at least 2 dimensions, got {x.shape}")

        # If last dim is not in_features but some other dim is, try to fix
        if x.shape[-1] != self.in_features:
            if x.shape[1] == self.in_features:
                # Possibly (B, in_features, ...)
                x = x.transpose(1, -1).contiguous()
            else:
                raise ValueError(f"Expected last or second dim to be in_features={self.in_features}, got {x.shape}")

        return super().forward(x)