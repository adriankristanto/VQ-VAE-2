import torch
import torch.nn as nn
from Encoders import TopEncoder, BottomEncoder
from VectorQuantizer import VectorQuantizerEMA
from Decoders import TopDecoder, BottomDecoder

class VQVAE2(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, num_resblocks, res_channels, D, K, beta=0.25, gamma=0.99):
        super(VQVAE2, self).__init__()
        # according to the figure 2a in the paper, 
        ################ encoders
        # input: output of bottom_encoder, hidden_channels
        # output: hidden_channels
        self.top_encoder = TopEncoder(hidden_channels, hidden_channels, num_resblocks, res_channels)
        # input: the input image with in_channels
        # output: hidden_channels
        self.bottom_encoder = BottomEncoder(in_channels, hidden_channels, num_resblocks, res_channels)
        ################ pre-quantizers
        # input: output of top_encoder, hidden_channels
        # output: D, to make sure that the number of channels equals to D (embedding dimension)
        self.top_pre_vq = nn.Conv2d(in_channels=hidden_channels, out_channels=D, kernel_size=1, stride=1)
        # input: output of top_decoder, D + output of bottom_encoder, hidden_channels
        # output: D, to make sure that the number of channels equals to D (embedding dimension)
        self.bottom_pre_vq = nn.Conv2d(in_channels=hidden_channels+D, out_channels=D, kernel_size=1, stride=1)
        ################ quantizer
        self.vectorquantizer = VectorQuantizerEMA(D, K, beta, gamma)
        ################ post-quantizer
        # input: top quantized, D
        # output: D
        self.top_post_vq = nn.ConvTranspose2d(in_channels=D, out_channels=D, kernel_size=4, stride=2, padding=1)
        ################ decoders
        # input: output of post vq, D + bottom quantized, D
        # output: input image channel, in_channels
        self.top_decoder = TopDecoder(D, hidden_channels, num_resblocks, res_channels, D)
        self.bottom_decoder = BottomDecoder(D + D, hidden_channels, num_resblocks, res_channels, in_channels)
