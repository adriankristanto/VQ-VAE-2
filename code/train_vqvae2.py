import sys
import os
# # to avoid module not found error when VQVAE2 imports Encoders module
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/models/')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from models.VQVAE2 import VQVAE2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}\n", flush=True)

# directory setup
MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
if 'data' not in os.listdir(MAIN_DIR):
    print('creating data directory...', flush=True)
    os.mkdir(MAIN_DIR + 'data')
if 'generated_images' not in os.listdir(MAIN_DIR):
    print('creating generated_images directory...', flush=True)
    os.mkdir(MAIN_DIR + 'generated_images')
if 'logs' not in os.listdir(MAIN_DIR):
    print('creating logs directory...', flush=True)
    os.mkdir(MAIN_DIR + 'logs')
if 'reconstructed_images' not in os.listdir(MAIN_DIR):
    print('creating reconstructed_images directory...', flush=True)
    os.mkdir(MAIN_DIR + 'reconstructed_images')
if 'saved_models' not in os.listdir(MAIN_DIR):
    print('creating saved_models directory...', flush=True)
    os.mkdir(MAIN_DIR + 'saved_models')

# 1. load the data
ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
BATCH_SIZE = 128

train_transform = transforms.Compose([
    # the dataset to be used will be the FFHQ dataset of size (1024, 1024)
    # since this is a 2-level VQVAE2, we need to reshape the input image to shape (256, 256)
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.ImageFolder(root=ROOT_DIR, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

print(f"""
Total data: {len(trainset)}
""", flush=True)

# input image sample
# data_iter = iter(trainset)
# img = next(data_iter)[0]
# torchvision.utils.save_image(img, 'sample.png')

# 2. instantiate the model
net = VQVAE2(
    in_channels=3,
    hidden_channels=128,
    num_resblocks=2,
    res_channels=32,
    D=64,
    K=512,
    beta=0.25,
    gamma=0.99
)

print(f"{net}\n", flush=True)

multigpu = False
if torch.cuda.device_count() > 1:
    print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
    net = nn.DataParallel(net)
    multigpu = True

net.to(device)

# 3. define the loss function
# this is the first loss term, which will be optimized by both the encoder and the decoder
# the third loss term, which will be optimized by the encoder, will be returned by the vq layer
# the second loss term will not be used here, as EMA update is used.
reconstruction_loss = nn.MSELoss()

# 4. define the optimiser
# the learning rate used in the original implementation by the author
LEARNING_RATE = 3e-4
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)