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
        # since the weights of the top and bottom quantizer will be different, make sure to create 2
        # vector quantizer layers
        self.top_vectorquantizer = VectorQuantizerEMA(D, K, beta, gamma)
        self.bottom_vectorquantizer = VectorQuantizerEMA(D, K, beta, gamma)
        ################ post-quantizer
        # input: top quantized, D
        # output: D
        self.top_post_vq = nn.ConvTranspose2d(in_channels=D, out_channels=D, kernel_size=4, stride=2, padding=1)
        ################ decoders
        # input: output of post vq, D + bottom quantized, D
        # output: input image channel, in_channels
        self.top_decoder = TopDecoder(D, hidden_channels, num_resblocks, res_channels, D)
        self.bottom_decoder = BottomDecoder(D + D, hidden_channels, num_resblocks, res_channels, in_channels)
    
    def encode(self, x):
        # according to the paper,
        # the bottom encoder accepts the input image
        bottom_encoded = self.bottom_encoder(x)
        # the top encoder accepts the output of the bottom encoder
        top_encoded = self.top_encoder(bottom_encoded)
        # the output of the top encoder will then get passed on to the top vq layer
        top_encoded = self.top_pre_vq(top_encoded)
        top_quantized, top_loss, _, _, top_ids, _ = self.top_vectorquantizer(top_encoded)
        # next, top_quantized will be passed on to the top decoder
        top_decoded = self.top_decoder(top_quantized)
        # the output of the top decoder will then be concatenated with the output of the bottom encoder
        # on the dimension of the number of channels
        bottom_encoded = torch.cat([top_decoded, bottom_encoded], dim=1)
        # next, the concatenated tensors will be passed on to the bottom vq layer
        bottom_encoded = self.bottom_pre_vq(bottom_encoded)
        bottom_quantized, bottom_loss, _, _, bottom_ids, _ = self.bottom_vectorquantizer(bottom_encoded)
        return top_quantized, bottom_quantized, top_loss + bottom_loss, top_ids, bottom_ids
