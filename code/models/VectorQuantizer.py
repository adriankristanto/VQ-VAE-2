import torch
import torch.nn as nn 
import torch.nn.functional as F

# Sonnet's implementation: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
# Other references:
# https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
# NOTE: this implementation follows https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py closely

class VectorQuantizerEMA(nn.Module):

    """
    VectorQuantizer with Exponentially Moving Averages (EMA)

    In the paper, Exponentially Moving Averages (EMA) mentioned as an 
    alternative to the loss function to update the embedding vectors.
    
    Advantages:
        1. The EMA update method is independent of the optimiser used to train the model.
        2. Additionally, using this method results in faster training.
    """

    def __init__(self, D, K, beta=0.25, gamma=0.99, epsilon=1e-5):
        """
        According to the paper,
            D: dimensionality of each embedding vector, or embedding_dim in the sonnet's implementation
            K: the size of the discrete space (the number of embedding vectors), or num_embeddings in the sonnet's implementation
            beta: the hyperparameter that acts as a weighting to the lost term, or commitment_cost in the sonnet's implementation
                recommendation from the paper, beta=0.25
            gamma: controls the speed of the EMA, or decay in the sonnet's implementation
                recommendation from the paper, gamma=0.99
            epsilon: to avoid numerical instability (such as division by zero)
                from the original implementation, epsilon=1e-5
        """
        super(VectorQuantizerEMA, self).__init__()
        # assign the parameters to self
        self.D = D
        self.K = K
        # in this implementation, the loss will be multiplied by beta during the forward pass of the VQ layer
        # instead of during training
        # so that, during training, we can simply add up the latent loss and the reconstruction loss
        # without having to multiply to latent loss with beta

        # reference: https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
        # the above implementation return the latent loss without multiplying it with beta
        # it only multiplies the latent loss with beta during training
        # the original sonnet implementation, however, multiplied the latent loss with beta during the forward pass
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

        # create the embedding layer's weights
        # initialise the weights using randn
        embedding_init = torch.randn(D, K)
        # why use register_buffer? because we want to save the embedding weights, etc. into saved state
        # for training continuation
        # why not use nn.Embedding or register_parameter? here, we update the weights using EMA instead of the traditional update
        self.register_buffer("embedding", embedding_init)

        # the followings are required for the EMA computation and the weights update
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", embedding_init.clone())
    
    def quantize(self, encoding_indices):
        # since we created the embedding weights of shape (D, K), we need to transpose it to (K, D)
        # this is because nn.Embedding uses dimension (K, D)
        # .transpose(0, 1) will swap the first 2 dimension
        return F.embedding(encoding_indices, self.embedding.transpose(0, 1))

    def forward(self, x):
        # note: pytorch data shape == (batch size, channel, height, width)
        # here, we change the shape into (batc size, height, width, channel)
        # this is to make the shape follows the requirement of the next step
        x = x.permute(0, 2, 3, 1).contiguous()
        # note: contiguous() needs to be called after permute. Otherwise, the following view() will return an error
        
        # firstly, flatten all other dimension except for the last one (the channels dimension)
        # for example, in the sonnet documentation, input tensor of shape (16, 32, 32, 64)
        # will be reshaped to (16 * 32 * 32, 64)
        # which means we have 16 * 32 * 32 tensors of 64 dimensions
        # 64 here is the input parameter D in our implementation
        x_flatten = x.view(-1, self.D)

        # get the nearest embedding
        # by computing the distance between the encoder output and all embedding vector using the following formula
        # (z_ex - e_j)**2
        # where z_ex is the input/the encoding output and e_j is the embedding vector j
        # print(x_flatten.shape) # torch.Size([16384, 64])
        # print(self.embedding.shape) # torch.Size([64, 512])
        distance = (
            torch.sum(x_flatten ** 2, dim=1, keepdim=True) + # shape: torch.Size([16384, 1])
            torch.sum(self.embedding ** 2, dim=0, keepdim=True) - # shape: torch.Size([1, 512])
            # torch.Size([16384, 1]) + torch.Size([1, 512]), using python broadcasting: torch.Size([16384, 512]) + torch.Size([16384, 512])
            2 * torch.matmul(x_flatten, self.embedding) # shape: torch.Size([16384, 512])
        )
        # print(distance.shape) # torch.Size([16384, 512])
        # compute the distance between each of the 16384 input vector and 512 embedding vectors

        # for each of 16384 input vectors of size D, get the minimum distance
        # thus, out of K embedding vectors, choose 1 that gives us the minimum distance
        nearest_embedding_ids = torch.argmin(distance, dim=1)
        # print(nearest_embedding.shape) # torch.Size([16384])

        # get the nearest embedding vector from the nearest embedding indices that we obtained above
        # for all 16384 indices, we get the corresponding vector from the look up operation,
        # therefore, we get 16384 vectors of size 64, i.e. (16384, 64)
        # then, we unflatten it to make the shape of the output == the shape of the input
        quantized = self.quantize(nearest_embedding_ids).view(*x.shape)
        # print(quantized.shape) # torch.Size([16, 32, 32, 64])

        # create one-hot encoding where each row represents an array of size 512
        # the encoding will be used for EMA calculation
        encodings = F.one_hot(nearest_embedding_ids, num_classes=self.K).type(x_flatten.dtype)
        # print(encodings.shape) # torch.Size([16384, 512])

        # if currently training, update the weights via EMA calculation
        # according to the paper, there are 3 items that we need to update
        # NOTE: the EMA formula is 
        # gamma * previous_term + (1 - gamma) * the current term
        if self.training:
            # the first one is N, or the cluster size
            # NOTE: in the paper, n_i^(t) is the number of input vectors that are quantized into a specific 
            # embedding vector
            # essentially, the following is the number of input vectors represented by a specific embedding vector
            self.cluster_size = self.cluster_size * self.gamma + (1 - self.gamma) * torch.sum(encodings, dim=0)
            # print(torch.sum(encodings, dim=0).shape) # torch.Size([512])
            # torch.sum(encodings, dim=0) means how many input vectors correspond to each index out of 512
            # for example, if the second entry is 32, this means that out of 16384 input vectors, 32 of them are
            # represented by the second embedding vector

            # the second update is m
            # print(x_flatten.transpose(0, 1).shape) # torch.Size([64, 16384])
            # the output shape: (D, K)
            # I think here, we get the value for each of the 512 vectors for the corresponding vector update
            # i.e. column 1 will be used to update the weight of the embedding layer in column 1
            # however, this has not been normalised yet by the number of input vectors that is represented by the 
            # aforementioned embedding vector
            # essentially, the following is the new, unnormalised value of a specific embedding vector
            self.embed_avg = self.embed_avg * self.gamma + (1 - self.gamma) * torch.matmul(x_flatten.transpose(0, 1), encodings)

            # according to https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
            # the following is the laplace smoothing of the cluster size
            # which I believe is to remove the occurence of 0 in the cluster size
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon) / (n + self.K * self.epsilon) * n
            )
            
            # final update: the weights of the embedding layers
            self.embedding = self.embed_avg / cluster_size
            # print(self.embed_avg.shape) # torch.Size([64, 512])
            # print(cluster_size.shape) # torch.Size([512])

        # in the paper, the following is the third loss term that needs to be optimized by the encoder
        # beta * (x - stop_gradient(embedding)) ** 2
        loss = self.beta * torch.mean((x - quantized.detach()) ** 2)

        # the following is the straigh-through estimator
        # where the gradients from the decoder input are copied to the encoder output
        quantized = x + (quantized - x).detach()

        # formula to compute perplexity according to the original implementation:
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        # the original sonnet implementation returns :
        # quantized, loss, perplexity, encodings, encodings_indices and the distances
        return (
            # the output is of shape (batch size, height, width, channel)
            # change it to (batch size, channel, height, width), which is what pytorch uses
            quantized.permute(0, 3, 1, 2).contiguous(), 
            loss, 
            perplexity, 
            encodings, 
            # reshaped following the original implementation
            # e.g. from 16384 to (16, 32, 32)
            nearest_embedding_ids.view(*x.shape[:-1]), 
            distance
        )

if __name__ == "__main__":
    tensor = torch.randn((16, 64, 32, 32))
    net = VectorQuantizerEMA(D=64, K=512)
    print(net)
    print(net(tensor)[0].shape)
    print(net(tensor)[4].shape)