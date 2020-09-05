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
NUM_WORKERS = 0

train_transform = transforms.Compose([
    # the dataset to be used will be the FFHQ dataset of size (1024, 1024)
    # since this is a 2-level VQVAE2, we need to reshape the input image to shape (256, 256)
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.ImageFolder(root=ROOT_DIR, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

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
    res_channels=64,
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

# 5. train the model
MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
RECONSTRUCTED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../reconstructed_images/'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'vqvae2-model-epoch10.pth'
# in the paper, the authors did 304741 updates with 70000 images from the ffhq dataset
# since they were using 128 as the batch size, 
# 70000 // 128 + 1 = 547 updates per epoch
# 304741 // 547 + 1 = 558 epochs -> round up to 560
EPOCH = 560
SAVE_INTERVAL = 10
# for reconstruction test
RECONSTRUCTION_SIZE = 64

def save_training_progress(epoch, net, optimizer, filename):
    torch.save({
        # since the currect epoch has been completed, save the next epoch
        'epoch' : epoch + 1,
        'net_state_dict' : net.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
    }, filename)

next_epoch = 0
if CONTINUE_TRAIN:
    checkpoint = torch.load(CONTINUE_TRAIN_NAME)
    net.load_state_dict(checkpoint.get('net_state_dict'))
    optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))
    next_epoch = checkpoint.get('epoch')

# training loop
# wrap the trainloader in tqdm
trainloader = tqdm(trainloader)
for epoch in range(next_epoch, EPOCH):
    # running_loss = 0.0

    sample = None
    sampling = True

    net.train()
    for train_data in trainloader:
        inputs = train_data[0].to(device)
        # 1. zeroes the gradients
        # optimizer.zero_grad() vs net.zero_grad()
        # reference: https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426
        net.zero_grad()
        # 2. forward propagation
        outputs, commitment_loss = net(inputs)
        # 3. compute loss
        # note: the commitment loss has been multiplied by beta in the vq layer
        if multigpu:
            # if we are using multiple gpus for training, there will be x number of loss returned as a tensor 
            # instead of just a scalar
            # where x is the number of gpus that we used for training
            # this can cause an issue with the autograd
            # therefore, we need to make it a scalar by taking the mean
            commitment_loss = torch.mean(commitment_loss)
        mse = reconstruction_loss(outputs, inputs)
        loss = mse + commitment_loss
        # 4. backward propagation
        loss.backward()
        # 5. update parameters
        optimizer.step()

        # running_loss += loss.item()

        # add description to the wrapped trainloader
        trainloader.set_description((
            f"epoch: {epoch+1}/{EPOCH}; "
            f"mse: {mse.item():.5f}; "
            f"latent: {commitment_loss.item():.5f}; "
            f"loss: {loss.item():.5f}"
        ))

        if sampling:
            sampling = False
            # take the first RECONSTRUCTION_SIZE images for reconstruction testing
            sample = inputs[:RECONSTRUCTION_SIZE]
    
    # reconstruction test
    net.eval()
    with torch.no_grad():
        outputs, _ = net(sample)
        # unnormalise
        # reference: https://discuss.pytorch.org/t/understanding-transform-normalize/21730
        sample = sample * 0.5 + 0.5
        outputs = outputs * 0.5 + 0.5
        torchvision.utils.save_image(sample, RECONSTRUCTED_DIRPATH + f'vqvae2_real_{epoch+1}.png')
        torchvision.utils.save_image(outputs, RECONSTRUCTED_DIRPATH + f'vqvae2_reconstructed_{epoch+1}.png')
    
    # save the model
    if (epoch + 1) % SAVE_INTERVAL == 0:
        save_training_progress(epoch, net, optimizer, MODEL_DIRPATH + f'vqvae2-model-epoch{epoch + 1}.pth')
    
    # print(f"Training loss: {running_loss / len(trainloader)}", flush=True)

# save the model
save_training_progress(epoch, net, optimizer, MODEL_DIRPATH + f'vqvae2-model-epoch{epoch + 1}.pth')