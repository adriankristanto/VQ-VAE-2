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
        # the ids are required to train the PixelCNN model, which is the PixelSNAIL
        return top_quantized, bottom_quantized, top_loss + bottom_loss, top_ids, bottom_ids
    
    def decode(self, top_quantized, bottom_quantized):
        # according to the paper, 
        # the top_quantized is passed on to convtranspose layer for it to be able to be concatenated with
        # bottom_quantized
        # make sure that the (h, w) of the top_quantized == (h, w) of the bottom_quantized
        top_quantized = self.top_post_vq(top_quantized)
        # concatenate top_quantizeed and bottom_quantized
        bottom_quantized = torch.cat([top_quantized, bottom_quantized], dim=1)
        # decode the concatenated tensors
        bottom_decoded = self.bottom_decoder(bottom_quantized)
        return bottom_decoded
    
    def decode_latent(self, top_ids, bottom_ids):
        # note that this function differs from the decode() function above.
        # this function can be used for the generation part.
        # it doesn't accept the quantized code.
        # instead, it accepts the latent code, which corresponds to nearest_embedding_ids in VectorQuantizer.py.
        # once PixelCNN is implemented to generate the latent code (i.e. nearest_embedding_ids) from each level of VQVAE2,
        # we can use this function to generate a new image
        # in the paper, the implementation is shown in Figure 2b

        # firstly, pass the top latent code to the top VQ layer
        # Note that using FFHQ, where each image is transformed to (256, 256),
        # the top latent code would be of shape (batch_size, 32, 32, 1) or (batch_size, 32, 32)
        # once we quantize it, it will be of shape (batch_size, 32, 32, D) where D is the dimension of 
        # each embedding vector
        top_quantized = self.top_vectorquantizer.quantize(top_ids)
        # since pytorch needs the shape to be in the form of (batch_size, c, h, w) and top_quantized shape
        # is in the form of (batch_size, h, w, c), we need to change it
        top_quantized = top_quantized.permute(0, 3, 1, 2).contiguous()

        # next, we need to quantize the bottom latent code
        # Note that using FFHQ, where each image is transformed to (256, 256),
        # the bottom latent code would be of shape (batch_size, 64, 64, 1) or (batch_size, 64, 64)
        # once we quantize it, it will be of shape (batch_size, 64, 64, D) where D is the dimension of 
        # each embedding vector
        bottom_quantized = self.bottom_vectorquantizer.quantize(bottom_ids)
        # note that when creating the bottom latent code, we need to condition it on the top latent code
        # however, when decoding it, we don't need to condition anything on anything
        # similar to the top_quantized above, we need to change the shape of bottom_quantized
        bottom_quantized = bottom_quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        top_quantized, bottom_quantized, commitment_loss, _, _ = self.encode(x)
        x = self.decode(top_quantized, bottom_quantized)
        return x, commitment_loss

if __name__ == "__main__":
    net = VQVAE2(
        in_channels=3,
        hidden_channels=128,
        num_resblocks=2,
        res_channels=64,
        D=64,
        K=512
    )
    print(net)

    x = torch.randn((1, 3, 256, 256))
    print(net(x)[0].shape)