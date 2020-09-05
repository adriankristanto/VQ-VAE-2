import torch
import torch.nn as nn 

# Author's original implementation: https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
# Other references:
# https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py

class ResidualBlock(nn.Module):

    """
    According to the VQ-VAE paper, both the encoder and decoder have residual 3x3 blocks, 
    which are implemented as ReLU, 3x3 Conv, ReLU, 1x1 Conv
    """

    def __init__(self, in_channels, res_channels, hidden_channels):
        """
            1. in_channels is the number of input channels to the first conv layer, 
            2. res_channels is the number of output channels of the first conv layer 
                and the number of input channels to the second conv layer
            3. hidden_channels is the number of output channels of the second conv layer.
        """
        super(ResidualBlock, self).__init__()
        self.resblock = self._build(in_channels, res_channels, hidden_channels)
    
    def _build(self, in_channels, res_channels, hidden_channels):
        resblock = nn.Sequential(
            nn.ReLU(inplace=True),
            # the conv layer should use the same padding, thus,
            # NOTE: in == out == n
            # padding = (s(n - 1) - n + f) / 2
            # ((n - 1) -n + 3) / 2 = 1
            nn.Conv2d(in_channels=in_channels, out_channels=res_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            # padding = (s(out - 1) - n + f) / 2
            # ((n - 1) -n + 1) / 2 = 0
            nn.Conv2d(in_channels=res_channels, out_channels=hidden_channels, kernel_size=(1, 1), stride=1, padding=0)
        )
        return resblock

    def forward(self, x):
        out = self.resblock(x)
        out += x
        # each residual block doesn't wrap (res_x + x) with an activation function
        # as the next block implement ReLU as the first layer
        return out


if __name__ == "__main__":
    net = ResidualBlock(3, 32, 128)
    print(net)