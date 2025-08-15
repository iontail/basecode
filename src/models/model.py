import torch
import torch.nn as nn
import numpy as np

from typing import List
from base_models import BaseModel


class ResidualLayer(nn.Module):
    """
    Residual block with two convolutional layers and skip connection
    """
    def __init__(self, channels: int):
        """
        Initialize ResidualLayer
        Args:
            - channels: number of input/output channels
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block
        Args:
            - x: input tensor (batch_size, channels, height, width)
        Returns:
            - output: residual output tensor (batch_size, channels, height, width)
        """
        res = x
        x = self.block1(x)
        x = self.block2(x)
        return x + res


class Encoder(nn.Module):
    """
    Encoder block with residual layers followed by downsampling
    """
    def __init__(self, in_ch: int, out_ch: int, num_res_blocks: int):
        """
        Initialize Encoder
        Args:
            - in_ch: number of input channels
            - out_ch: number of output channels
            - num_res_blocks: number of residual blocks
        """
        super().__init__()
        self.res_blocks = nn.Sequential(*[ResidualLayer(in_ch) for _ in range(num_res_blocks)])
        self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder
        Args:
            - x: input tensor (batch_size, in_ch, height, width)
        Returns:
            - output: downsampled tensor (batch_size, out_ch, height//2, width//2)
        """
        x = self.res_blocks(x)
        x = self.downsample(x)
        return x


class Midcoder(nn.Module):
    """
    Middle processing block with residual layers
    """
    def __init__(self, channels: int, num_res_blocks: int):
        """
        Initialize Midcoder
        Args:
            - channels: number of channels
            - num_res_blocks: number of residual blocks
        """
        super().__init__()
        self.res_blocks = nn.Sequential(*[ResidualLayer(channels) for _ in range(num_res_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through midcoder
        Args:
            - x: input tensor (batch_size, channels, height, width)
        Returns:
            - output: processed tensor (batch_size, channels, height, width)
        """
        return self.res_blocks(x)


class Decoder(nn.Module):
    """
    Decoder block with upsampling followed by residual layers
    """
    def __init__(self, in_ch: int, out_ch: int, num_res_blocks: int):
        """
        Initialize Decoder
        Args:
            - in_ch: number of input channels
            - out_ch: number of output channels
            - num_res_blocks: number of residual blocks
        """
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.res_blocks = nn.Sequential(*[ResidualLayer(out_ch) for _ in range(num_res_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder
        Args:
            - x: input tensor (batch_size, in_ch, height, width)
        Returns:
            - output: upsampled tensor (batch_size, out_ch, height*2, width*2)
        """
        x = self.upsample(x)
        x = self.res_blocks(x)
        return x

    
class UNet(BaseModel):
    """
    U-Net architecture with residual blocks for image-to-image tasks
    """
    def __init__(self, channels: List[int], num_res_blocks: int, *args, **kwargs):
        """
        Initialize UNet
        Args:
            - channels: list of channel sizes for each encoder/decoder level
            - num_res_blocks: number of residual blocks per level
            - *args: additional arguments for BaseModel
            - **kwargs: additional keyword arguments for BaseModel
        """
        super().__init__(*args, **kwargs)

        self.init_conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )

        self.encoders = nn.ModuleList([
            Encoder(in_ch, out_ch, num_res_blocks)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        ])
        self.decoders = nn.ModuleList([
            Decoder(in_ch, out_ch, num_res_blocks)
            for in_ch, out_ch in zip(reversed(channels[1:]), reversed(channels[:-1]))
        ])

        self.midcoder = Midcoder(channels[-1], num_res_blocks)
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)
        
        self.hidden_dim = channels[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net
        Args:
            - x: input tensor (batch_size, 1, height, width)
        Returns:
            - output: reconstructed tensor (batch_size, 1, height, width)
        """
        x = self.init_conv(x)
        skips = []

        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        x = self.midcoder(x)

        for decoder in self.decoders:
            skip = skips.pop()
            x = x + skip
            x = decoder(x)

        return self.final_conv(x)
    
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference pass through U-Net
        Args:
            - x: input tensor (batch_size, 1, height, width)
        Returns:
            - output: reconstructed tensor (batch_size, 1, height, width)
        """
        return self.forward(x)
    
    @property
    def dim(self) -> int:
        """
        Get model hidden dimension
        Returns:
            - dim: hidden dimension of the model
        """
        return self.hidden_dim
    

if __name__ == "__main__":
    model = UNet(channels=[32, 64, 128], num_res_blocks=2)
    model.model_summary()
    
    x = torch.randn(2, 1, 32, 32)
    
    with torch.no_grad():
        output = model.inference(x)

    print(output.shape)
    print(model.dim)