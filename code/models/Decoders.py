import torch
import torch.nn as nn
from Residual import ResidualBlock

# references: 
# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
# https://github.com/unixpickle/vq-vae-2

class BottomDecoder(nn.Module):

    """
    Similar to the bottom encoder, the bottom decoder has the same architecture as 
    the decoder used in VQ-VAE-1.
    It will upsample the input by a factor of 4.
    For example, given an input of shape (64, 64, hidden_channels) => (256, 256, 3)
    """

    def __init__(self, in_channels, hidden_channels, num_resblocks, res_channels, out_channels):
        super(BottomDecoder, self).__init__()
        self.layers = self._build(in_channels, hidden_channels, num_resblocks, res_channels, out_channels)
    
    def _build(self, in_channels, hidden_channels, num_resblocks, res_channels, out_channels):
        layers = [
            # here, we want to use the same padding to keep the output size the same as the input size
            # padding = (s(n - 1) - n + f) / 2
            # ((n - 1) -n + 3) / 2 = 1
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        ]
        layers += [
            # here, we create num_resblocks number of residual blocks
            ResidualBlock(hidden_channels, res_channels, hidden_channels) for _ in range(num_resblocks)
        ]
        layers += [
            # each resblock output is not wrapped by the activation function as the next block has ReLU as its first layer
            # however, the final resblock doesn't have anything to wrap its output with an activation function
            # therefore, here we wrap the output of the final resblock with ReLU
            nn.ReLU(inplace=True)
        ]
        layers += [
            # p = (2(in - 1) + 4 - 2in) / 2 = 1
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # p = (2(in - 1) + 4 - 2in) / 2 = 1
            nn.ConvTranspose2d(in_channels=hidden_channels//2, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class TopDecoder(nn.Module):

    """
    The top decoder will upsample the input by a factor of 2.
    For example, given an input of shape (32, 32, hidden_channels) => (64, 64, hidden_channels)
    """

    def __init__(self, in_channels, hidden_channels, num_resblocks, res_channels, out_channels):
        super(TopDecoder, self).__init__()
        self.layers = self._build(in_channels, hidden_channels, num_resblocks, res_channels, out_channels)
    
    def _build(self, in_channels, hidden_channels, num_resblocks, res_channels, out_channels):
        layers = [
            # here, we want to use the same padding to keep the output size the same as the input size
            # padding = (s(n - 1) - n + f) / 2
            # ((n - 1) -n + 3) / 2 = 1
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        ]
        layers += [
            # here, we create num_resblocks number of residual blocks
            ResidualBlock(hidden_channels, res_channels, hidden_channels) for _ in range(num_resblocks)
        ]
        layers += [
            # each resblock output is not wrapped by the activation function as the next block has ReLU as its first layer
            # however, the final resblock doesn't have anything to wrap its output with an activation function
            # therefore, here we wrap the output of the final resblock with ReLU
            nn.ReLU(inplace=True)
        ]
        layers += [
            # p = (2(in - 1) + 4 - 2in) / 2 = 1
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == "__main__":
    topdecoder = TopDecoder(128, 128, 2, 32, 128)
    bottomdecoder = BottomDecoder(128 + 128, 128, 2, 32, 3)

    top_x = torch.randn((1, 128, 32, 32))
    bottom_x = torch.randn((1, 128, 64, 64))

    top_y = topdecoder(top_x)
    print(f"top_y shape: {top_y.shape}") # top_y shape: torch.Size([1, 128, 64, 64])
    x = torch.cat([top_y, bottom_x], dim=1)
    print(f"concatenate top_y with bottom_x shape: {x.shape}") # concatenate top_y with bottom_x shape: torch.Size([1, 256, 64, 64])

    # decoding process
    upsample_top_x = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)(top_x)
    print(f'upsample_top_x shape: {upsample_top_x.shape}')
    bottom_y = torch.cat([upsample_top_x, bottom_x], dim=1)
    print(f'concatenate top_y with bottom_x shape: {bottom_y.shape}')
    y = bottomdecoder(bottom_y)
    print(f'output shape: {y.shape}')

