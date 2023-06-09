# -*- coding: utf-8 -*-
"""noise.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gt1IavYvS2k_MGQcdRFApcFkyBKfruZa
"""

# Training progress bar
!pip install -q qqdm
import math
import cv2
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, BatchSampler
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW
from qqdm import qqdm, format_str
import pandas as pd
import matplotlib.pyplot as plt

# AE model
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1),         
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=1, padding=1),        
            nn.ReLU(),
			      nn.Conv2d(24, 48, 3, stride=1, padding=1),         
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			      nn.ConvTranspose2d(48, 24, 3, stride=1, padding=1),
            nn.ReLU(),
			      nn.ConvTranspose2d(24, 12, 3, stride=1, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1),          
            nn.ReLU(), 
            nn.Conv2d(12, 24, 3, stride=1, padding=1),           
            nn.ReLU(), 
            nn.Conv2d(24, 48, 3, stride=1, padding=1),           
            nn.ReLU()
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=1, padding=1),  
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(96, 48, 3, stride=1, padding=1),           
            nn.ReLU(), 
            nn.ConvTranspose2d(48, 24, 3, stride=1, padding=1), 
            nn.ReLU(),
			      nn.ConvTranspose2d(24, 12, 3, stride=1, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 3, stride=1, padding=1), 
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

# Load original image & label
original_image = np.load("eye/data.npy")
original_label = np.load("eye/label.npy")
input = torch.from_numpy(original_image*2-1).permute(0,3,1,2).float().cuda()

# Load in AE & VAE model weight
best_model_AE = torch.load("best_model_cnn.pth")
best_model_VAE = torch.load("best_model_vae.pth")

# Compute latent tensor of AE & VAE
idx = [0,1,2,3,4,225,226,227,228,229,840,841,842,843,844,1470,1471,1472,1473,1474]
latent_AE = best_model_AE.encoder(input)[idx].repeat_interleave(5,dim=0)
latent_mu = best_model_VAE.encode(input)[0][idx]
latent_logvar = best_model_VAE.encode(input)[1][idx]
latent_VAE = best_model_VAE.reparametrize(latent_mu,latent_logvar).repeat_interleave(5,dim=0)
gen_label = original_label[idx].repeat(5)

# Adding Gaussian noise into latent tensor
std = 0.1
latent_AE += std*torch.randn(100,48,50,50).cuda()
latent_VAE += std*torch.randn(100,96,50,50).cuda()

# Generate image of AE & VAE
output_AE = best_model_AE.decoder(latent_AE).cpu().permute(0,2,3,1)
output_AE = ((output_AE+1)/2).detach().numpy()
output_VAE = best_model_VAE.decode(latent_VAE).cpu().permute(0,2,3,1)
output_VAE = ((output_VAE+1)/2).detach().numpy()

# Visualization
plt.subplot(2,3,1)
plt.imshow(original_image[2]) # gen id 3 of VAE
plt.title("Original")
for i in range(5):
  plt.subplot(2,3,2+i)
  plt.imshow(output_VAE[10+i])
plt.savefig("gen_VAE_id3", dpi = 300)
plt.show()

plt.subplot(2,3,1)
plt.imshow(original_image[2]) # gen id 3 of AE
plt.title("Original")
for i in range(5):
  plt.subplot(2,3,2+i)
  plt.imshow(output_AE[10+i])
plt.savefig("gen_AE_id3", dpi = 300)
plt.show()

plt.subplot(2,3,1)
plt.imshow(original_image[226]) # gen id 227 of VAE
plt.title("Original")
for i in range(5):
  plt.subplot(2,3,2+i)
  plt.imshow(output_VAE[30+i])
plt.savefig("gen_VAE_id227", dpi = 300)
plt.show()

plt.subplot(2,3,1)
plt.imshow(original_image[226]) # gen id 227 of AE
plt.title("Original")
for i in range(5):
  plt.subplot(2,3,2+i)
  plt.imshow(output_AE[30+i])
plt.savefig("gen_AE_id227", dpi = 300)
plt.show()

plt.subplot(2,3,1)
plt.imshow(original_image[840]) # gen id 841 of VAE
plt.title("Original")
for i in range(5):
  plt.subplot(2,3,2+i)
  plt.imshow(output_VAE[50+i])
plt.savefig("gen_VAE_id841", dpi = 300)
plt.show()

plt.subplot(2,3,1)
plt.imshow(original_image[840]) # gen id 841 of AE
plt.title("Original")
for i in range(5):
  plt.subplot(2,3,2+i)
  plt.imshow(output_AE[50+i])
plt.savefig("gen_AE_id841", dpi = 300)
plt.show()

plt.subplot(2,3,1)
plt.imshow(original_image[1474]) # gen id 1475 of VAE
plt.title("Original")
for i in range(5):
  plt.subplot(2,3,2+i)
  plt.imshow(output_VAE[95+i])
plt.savefig("gen_VAE_id1475", dpi = 300)
plt.show()

plt.subplot(2,3,1)
plt.imshow(original_image[1474]) # gen id 1475 of AE
plt.title("Original")
for i in range(5):
  plt.subplot(2,3,2+i)
  plt.imshow(output_AE[95+i])

plt.savefig("gen_AE_id1475", dpi = 300)
plt.show()

# Save gen_data and gen_label as .npy files
np.save("gen_data.npy", output_VAE)
np.save("gen_label.npy", gen_label)