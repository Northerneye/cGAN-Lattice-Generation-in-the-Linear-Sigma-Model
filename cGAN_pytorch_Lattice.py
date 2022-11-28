import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from convNd import convNd

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--run", type=int, default=0, help="0 for training, 1 for testing")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        # fully connected layer, output 10 classes
        self.linear = nn.Linear(101, 8192)

        self.weight1 = torch.rand(1)[0]
        self.bias1 = torch.rand(1)[0]

        self.conv1t = convNd(
            in_channels=1,
            out_channels=2,
            num_dims=4,
            kernel_size=3,
            stride=(1,1,1,1),
            padding=1,
            padding_mode='zeros',
            output_padding=0,
            is_transposed=True,
            use_bias=True,
            groups=1,
            kernel_initializer=lambda x: torch.nn.init.constant_(x, self.weight1), 
            bias_initializer=lambda x: torch.nn.init.constant_(x, self.bias1)
        )

        self.weight2 = torch.rand(1)[0]
        self.bias2 = torch.rand(1)[0]
        
        self.conv2t = convNd(
            in_channels=2,
            out_channels=4,
            num_dims=4,
            kernel_size=3,
            stride=(1,1,1,1),
            padding=1,
            padding_mode='zeros',
            output_padding=0,
            is_transposed=True,
            use_bias=True,
            groups=1,
            kernel_initializer=lambda x: torch.nn.init.constant_(x, self.weight2), 
            bias_initializer=lambda x: torch.nn.init.constant_(x, self.bias2)
        )        

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        #gen_input = torch.cat((self.label_emb(labels), noise), -1)
        label = torch.reshape(labels, (64, 1)) #(1, batch_size)
        gen_input = torch.cat((noise,label), dim=1)
        gen_input = self.linear(gen_input)

        gen_input = torch.reshape(gen_input, (64, 1, 8, 8, 8, 16))

        gen_input = self.conv1t(gen_input)
        gen_input = self.conv2t(gen_input)
        return gen_input


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        self.weight1 = torch.rand(1)[0]
        self.bias1 = torch.rand(1)[0]

        self.conv1 = nn.Sequential(         
            convNd(
                in_channels=4, 
                out_channels=2,
                num_dims=4, 
                kernel_size=3, 
                stride=(2,2,2,2), 
                padding=1, 
                padding_mode='zeros',
                output_padding=0,
                is_transposed=False,
                use_bias=True, 
                groups=1,
                kernel_initializer=lambda x: torch.nn.init.constant_(x, self.weight1), 
                bias_initializer=lambda x: torch.nn.init.constant_(x, self.bias1)
            ),                              
            nn.ReLU(),                      
            #nn.MaxPool2d(kernel_size=2),    #NEED MAX_POOLING_4D
        )

        self.weight2 = torch.rand(1)[0]
        self.bias2 = torch.rand(1)[0]

        self.conv2 = nn.Sequential(         
            convNd(
                in_channels=2, 
                out_channels=1,
                num_dims=4, 
                kernel_size=3, 
                stride=(2,2,2,2), 
                padding=1, 
                padding_mode='zeros',
                output_padding=0,
                is_transposed=False,
                use_bias=True, 
                groups=1,
                kernel_initializer=lambda x: torch.nn.init.constant_(x, self.weight2), 
                bias_initializer=lambda x: torch.nn.init.constant_(x, self.bias2)
            ),                              
            nn.ReLU(),                      
            #nn.MaxPool2d(kernel_size=2),    #NEED MAX_POOLING_4D
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32  + 1, 1)

    def forward(self, x, label):
        #gen_input = torch.cat((self.label_emb(labels), x), -1)
        gen_input = torch.reshape(x, (64,4,8,8,8,16))
        gen_input = self.conv1(gen_input)
        gen_input = self.conv2(gen_input)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        gen_input = torch.reshape(gen_input, (64, 32))#(batch_size, elements_per_image_at_this_point)
        label = torch.reshape(label, (64, 1)) #(1, batch_size)
        gen_input = torch.cat((gen_input,label), dim=1)
        output = self.out(gen_input)
        return output

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
generator.load_state_dict(torch.load("generator_epoch_30.pt"))
discriminator.load_state_dict(torch.load("discriminator_epoch_30.pt"))

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if(opt.run == 0):
    #load dataset
    class LatticeDataset(Dataset):
        def __init__(self):
            self.i = -1
            self.x_train = [] #x is field values
            self.y_train = [] #y is going to be alpha value
            files = os.listdir()
            print("Loading Field Files...")
            for file in files:
                if(".npy" in file):
                    datapoint = np.load(file)
                    datapoint = np.array([[[[[datapoint[j][k][l][m][i] for m in range(16)] for l in range(8)] for k in range(8)] for j in range(8)] for i in range(4)])
                    if(("alph_0.0015" not in file) and ("alph_0.0016" not in file)):
                        self.x_train.append(datapoint)
                        for i in [0.0011, 0.0012, 0.0013, 0.0014, 0.0017, 0.0018, 0.0019]:
                            if("alph_"+str(i) in file):
                                self.y_train.append([(i-0.001)*1000])
            self.x_train = np.array(self.x_train)
            self.y_train = np.array(self.y_train)
            print("Finished Loading...")
            print("x_train:")
            print(self.x_train.shape)
            print("y_train:")
            print(self.y_train.shape)

        def __len__(self):
            #return len(self.x_train)
            return 2816

        def __getitem__(self, idx):
            self.i = self.i + 1
            if(self.i == 2800):
                self.i = 0
            return torch.tensor(self.x_train[self.i]), torch.tensor(self.y_train[self.i])

    Lattices = LatticeDataset()

    dataloader = torch.utils.data.DataLoader(
        Lattices,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------
    epochs = 40
    epochs_done = 0
    for epoch in range(epochs):
        for i, (lattice, labels) in enumerate(dataloader):
            imgs = lattice
            
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        epochs_done = epochs_done + 1
        if epochs_done % 1 == 0:
            torch.save(generator.state_dict(), "generator_epoch_"+str(epochs_done)+".pt")
            torch.save(discriminator.state_dict(), "discriminator_epoch_"+str(epochs_done)+".pt")
else:#generate alpha=0.0015
    print("Generating Test Lattices")
    generator.load_state_dict(torch.load("generator_epoch_"+str(opt.run)+".pt"))
    for i in range(7):
        print("Working on batch "+str(i))
        batch_size=64
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.full((batch_size), 0.5))) 
        gen_imgs = generator(z, gen_labels)
        np_arr = gen_imgs.cpu().detach().numpy()
        for j in range(len(np_arr)):
            with open(f"cGAN_pions_{i*64+j}_8x16_msq_-20550.0_lmbd_100000.0_alph_0.0015.npy", 'wb') as f:
                np.save(f, np.array(np_arr[j]))
    print("Generating Validation Lattices")
    for i in range(7):
        print("Working on batch "+str(i))
        batch_size=64
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.full((batch_size), 0.6))) 
        gen_imgs = generator(z, gen_labels)
        np_arr = gen_imgs.cpu().detach().numpy()
        for j in range(len(np_arr)):
            with open(f"cGAN_pions_{i*64+j}_8x16_msq_-20550.0_lmbd_100000.0_alph_0.0016.npy", 'wb') as f:
                np.save(f, np.array(np_arr[j]))

        
