import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = 64,
        in_feature: int = 768):

        super().__init__()

        self.first_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels*2,
            kernel_size=(1, 100, 1), 
        )

        self.second_conv = nn.Conv3d(
            in_channels=out_channels*2,
            out_channels=out_channels,
            kernel_size=(1, 84, 1),
        )        

        self.third_conv = nn.Conv3d(
            in_channels=out_channels,
            out_channels=16,
            kernel_size=(1, 193, 1),
        )        
        
        self.flatten = nn.Flatten(2)
        self.fc = nn.Linear(1712, in_feature)


    def forward(self, x) -> torch.Tensor:
        """Apply forward pass."""
        x = x.unsqueeze(0)
        x = x.permute(1, 3, 2, 4, 0)

        x = self.first_conv(x)
        x = F.gelu(x)
        x = self.second_conv(x)
        x = F.gelu(x)
        x = self.third_conv(x)
        x = F.gelu(x)
        x = torch.swapaxes(x, 1, 2)
        x = self.flatten(x) # x shape after flattening --> torch.Size([8, 20, 1024])
        x = self.fc(x) # x shape after linear --> torch.Size([8, 20, 512])      
        return x