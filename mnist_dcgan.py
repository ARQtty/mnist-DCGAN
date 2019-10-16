import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


INPUTS_SIZE = 29*29
img_size = 29
BATCH_SIZE = 500


def plot(samples):
    fig = plt.figure(figsize=(3, 3))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_size, img_size), cmap='Greys_r')

    return fig


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)



class Dataset():
    def __init__(self):
        transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
        trainData = datasets.MNIST('./mnist_data', train=True, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(trainData,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=2)

    def trainLength(self):
        return 60000


class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        output_size = 1
        hidden_size = 16

        self.D = nn.Sequential(
            nn.Conv2d(1, hidden_size*2, 5, stride=2),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size*2, hidden_size*4, 5, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            View([-1, hidden_size*4]),

            nn.Linear(hidden_size*4, output_size),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        out = self.D(inputs.to(device))
        return out

    def eval(self, inputs):
        pass


class G_net(nn.Module):
    def __init__(self):
        super(G_net, self).__init__()
        self.s = 1

        self.z_dim = 100
        hidden_size = 16

        self.G = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(self.z_dim, hidden_size*4, kernel_size=4),
            nn.BatchNorm2d(hidden_size*4),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(hidden_size*4, hidden_size*2, kernel_size=4),
            nn.BatchNorm2d(hidden_size*2),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(hidden_size*2, 1, kernel_size=4),

            nn.Tanh()
        )

    def forward(self, dummy_for_summary=None):
        inputs = self.generate_noise((BATCH_SIZE, self.z_dim)).to(device)
        out = self.G(inputs)
        return out

    def generate_noise(self, size):
        noise = torch.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(*size, self.s, self.s)))
        return noise

    def sample(self):
        with torch.no_grad():
            inputs = self.generate_noise((9, self.z_dim))
            out = self.G(inputs.to(device))
            return out.cpu()

    def eval(self):
        pass


SAVE_SAMPLES = 1

epoches = 100
lossEvery = 10


device = torch.device("cuda:0")
D = D_net().to(device)
G = G_net().to(device)
dataloader = Dataset()

criterion = nn.BCELoss()
D_optimizer = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
G_optimizer = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))


# Train
for epoch in range(epoches):
    running_loss_D = 0.0
    running_loss_G = 0.0
    for i, data in enumerate(dataloader.trainloader, 0):

        inputs, labels = data
        if (inputs.size()[0] != BATCH_SIZE):
            continue
        inputs.to(device)

        # zero the parameter gradients
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()


        # inputs and ground truth data
        fake_inputs = G()
        valid = torch.ones(BATCH_SIZE).to(device)
        fake = torch.zeros(BATCH_SIZE).to(device)

        # train G_net
        G_loss = criterion(D(fake_inputs), valid)
        G_loss.backward()
        G_optimizer.step()

        # train D_net
        D_real = D(inputs).squeeze()
        D_fake = D(fake_inputs.detach()).squeeze()

        D_real_loss = criterion(D_real, valid)
        D_fake_loss = criterion(D_fake, fake)
        D_loss = D_real_loss.add(D_fake_loss)
        D_loss.backward()
        D_optimizer.step()

        running_loss_D += D_loss.item()
        running_loss_G += G_loss.item()

        # Save g output
        if i % 10 == 0 and SAVE_SAMPLES:
            samples = torch.reshape(G.sample(), (9, img_size, img_size)).detach()
            fig = plot(samples)
            plt.savefig('./out/{}.png'.format(str(epoch).zfill(3)+str(i).zfill(4)), bbox_inches='tight')
            plt.close(fig)

        # logging loss
        if i % lossEvery == 0:
            print("E %2d  s %3d     lossD: %.4f    lossG: %.4f" % (epoch, i, running_loss_D / lossEvery, running_loss_G / lossEvery))
            running_loss_D = 0.
            running_loss_G = 0.

    print("Epoch ended")
