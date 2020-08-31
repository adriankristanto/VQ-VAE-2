import torch
import torch.nn as nn
from Residual import ResidualBlock

# references: 
# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
# https://github.com/unixpickle/vq-vae-2

class BottomEncoder(nn.Module):

    """
    The bottom encoder has the same architecture as the encoder from VQ-VAE-1.
    It will downsample the input image by a factor of 4.
    For example, input image of shape (256, 256, 3) => output image of shape (64, 64, hidden_channels).
    """

    def __init__(self, in_channels, hidden_channels, num_resblocks, res_channels):
        super(BottomEncoder, self).__init__()
        self.layers = self._build(in_channels, hidden_channels, num_resblocks, res_channels)
    
    def _build(self, in_channels, hidden_channels, num_resblocks, res_channels):
        layers = [
            # padding = (2(n/2 - 1) -n + 4) / 2 = 1
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # padding = (2(n/2 - 1) -n + 4) / 2 = 1
            nn.Conv2d(in_channels=hidden_channels // 2, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # padding = ((n - 1) -n + 3) / 2 = 1
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        ]
        layers += [
            # here, we create num_resblocks number of residual blocks
            ResidualBlock(hidden_channels, res_channels, hidden_channels) for _ in range(num_resblocks)
        ]
        layers += [
            # each resblock output is not wrapped by the activation function as the next block has ReLU as its first layer
            # however, the final resblock doesn't have anything to wrap its output with an activation function
            # therefore, here we wrap the output of the final resblock with ReLU
            nn.ReLU()
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x