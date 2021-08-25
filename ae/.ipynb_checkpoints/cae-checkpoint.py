import torch
from torch import nn
import torch.nn.functional as F

class CAE(nn.Module):
    def __init__(self, in_shape, n_z):
        super(CAE, self).__init__()
        self.n_z = n_z
        self.in_shape = in_shape
        c, h, w = in_shape
        self.z_dim = h//2**2 
################################################################################
### Encoder ####################################################################

        self.conv_enc1 = nn.Conv2d(c, 6, kernel_size=3, padding=1)
        self.bn_enc1 = nn.BatchNorm2d(6)
        # conv1 MNIST: (6, 28, 28), CIFAR: (6, 32, 32)
        # pool1 MNIST: (6, 14, 14), CIFAR: (6, 16, 16)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.LeakyReLU()
        
        self.conv_enc2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.bn_enc2 = nn.BatchNorm2d(16)
        # conv2 MNIST: (16, 14, 14), CIFAR: (16, 16, 16)
        # pool2 MNIST: (16, 7, 7), CIFAR: (16, 8, 8)
################################################################################
### Latent Space ###############################################################

        self.fc_z = nn.Linear(16*self.z_dim**2, self.n_z)
        # linear MNIST: (784, 10), CIFAR: (1024, 10)
        self.fc_z_develop = nn.Linear(self.n_z, 16*self.z_dim**2)
        # linear MNIST: (10, 784), CIFAR: (10, 1024)
################################################################################
### Decoder ####################################################################

        self.conv_dec1 = nn.ConvTranspose2d(16, 6, kernel_size=2, stride=2)
        self.bn_dec1 = nn.BatchNorm2d(6)
        # dec1 MNIST: (6, 14, 14), CIFAR: (6, 16, 16)
        self.conv_dec2 = nn.ConvTranspose2d(6, c, kernel_size=2, stride=2)
        # dec2 MNIST: (16, 14, 14), CIFAR: (16, 16, 16)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        enc1 = self.pool(self.relu(self.bn_enc1(self.conv_enc1(x))))
        enc2 = self.pool(self.relu(self.bn_enc2(self.conv_enc2(enc1))))
        
        flat_enc2 = enc2.view(enc2.shape[0], -1)
        z = self.fc_z(flat_enc2)
        z_develop = self.relu(self.fc_z_develop(z))
        c, h, w = enc2[0].shape
        z_develop = z_develop.view(-1, c, h, w)
        
        dec1 = self.relu(self.bn_dec1(self.conv_dec1(z_develop)))
        x_bar = self.sig(self.conv_dec2(dec1))
        
        return x_bar, z

################################################################################
################################################################################
################################################################################
### Fully Convolutional AE #####################################################

class FCAE(nn.Module):
    def __init__(self, in_shape, n_z):
        super(FCAE, self).__init__()
        self.n_z = n_z
        self.in_shape = in_shape
        c, h, w = in_shape
        self.z_dim = h//2**2 - 3
################################################################################
### Encoder ####################################################################

        self.conv_enc1 = nn.Conv2d(c, 6, kernel_size=5, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.bn_enc1 = nn.BatchNorm2d(6)
        # conv1 (6, 24, 24)
        # pool1 (6, 12, 12)
        
        self.conv_enc2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.bn_enc2 = nn.BatchNorm2d(16)
        # conv2 (16, 8, 8)
        # pool2 (16, 4, 4)
################################################################################
### Latent Space ###############################################################        
        
        self.z_layer = nn.Conv2d(16, n_z, kernel_size=self.z_dim, padding=0)
        # conv2 (10, 1, 1)
        self.z_develop = nn.ConvTranspose2d(n_z, 16, kernel_size=self.z_dim, padding=0)
        self.bn_z_dev = nn.BatchNorm2d(16)
        # z_develop (16, 4, 4)
################################################################################
### Decoder ####################################################################
        self.unpool = nn.MaxUnpool2d(2, 2)
        #unpool2 (16, 8, 8)
        self.conv_dec2 = nn.ConvTranspose2d(16, 6, kernel_size=5, padding=0)
        self.bn_dec2 = nn.BatchNorm2d(6)
        # dec2 (6, 12, 12)
        # unpool1 (6, 24, 24)
        
        self.conv_dec1 = nn.ConvTranspose2d(6, c, kernel_size=5, padding=0)
        self.sig = nn.Sigmoid()
        # dec3 (1, 28, 28)
    
    def forward(self, x):
        enc1, idx1 = self.pool(self.bn_enc1(self.relu(self.conv_enc1(x))))
        enc2, idx2 = self.pool(self.bn_enc2(self.relu(self.conv_enc2(enc1))))
        
        z = self.z_layer(enc2)
        z_flat = z.view(z.shape[0], -1)
        dec2 = self.bn_z_dev(self.relu(self.z_develop(z)))
        
        dec1 = self.bn_dec2(self.relu(self.conv_dec2(self.unpool(dec2, idx2))))
        x_bar = self.sig(self.conv_dec1(self.unpool(dec1, idx1)))
        
        return x_bar, z_flat
    
################################################################################
################################################################################
################################################################################
### Convolutional AE with MaxUnpool ############################################

class CAEUnpool(nn.Module):
    def __init__(self, in_shape, n_z):
        super().__init__()
        self.n_z = n_z
        self.in_shape = in_shape
        c, h, w = in_shape
        self.z_dim = h//2**2 
################################################################################
### Encoder ####################################################################

        self.conv_enc1 = nn.Conv2d(c, 6, kernel_size=3, padding=1)
        self.bn_enc1 = nn.BatchNorm2d(6)
        # conv1 MNIST: (6, 28, 28), CIFAR: (6, 32, 32)
        # pool1 MNIST: (6, 14, 14), CIFAR: (6, 16, 16)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.relu = nn.LeakyReLU()
        
        self.conv_enc2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.bn_enc2 = nn.BatchNorm2d(16)
        # conv2 MNIST: (16, 14, 14), CIFAR: (16, 16, 16)
        # pool2 MNIST: (16, 7, 7), CIFAR: (16, 8, 8)
################################################################################
### Latent Space ###############################################################

        self.fc_z = nn.Linear(16*self.z_dim**2, self.n_z)
        # linear MNIST: (784, 10), CIFAR: (1024, 10)
        self.fc_z_develop = nn.Linear(self.n_z, 16*self.z_dim**2)
        # linear MNIST: (10, 784), CIFAR: (10, 1024)
################################################################################
### Decoder ####################################################################
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv_dec1 = nn.ConvTranspose2d(16, 6, kernel_size=3, padding=1)
        self.bn_dec1 = nn.BatchNorm2d(6)
        # dec1 MNIST: (6, 14, 14), CIFAR: (6, 16, 16)
        self.conv_dec2 = nn.ConvTranspose2d(6, c, kernel_size=3, padding=1)
        # dec2 MNIST: (16, 14, 14), CIFAR: (16, 16, 16)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        enc1, idx1 = self.pool(self.relu(self.bn_enc1(self.conv_enc1(x))))
        enc2, idx2 = self.pool(self.relu(self.bn_enc2(self.conv_enc2(enc1))))
        
        flat_enc2 = enc2.view(enc2.shape[0], -1)
        z = self.fc_z(flat_enc2)
        z_develop = self.relu(self.fc_z_develop(z))
        c, h, w = enc2[0].shape
        z_develop = z_develop.view(-1, c, h, w)
        
        dec1 = self.relu(self.bn_dec1(self.conv_dec1(self.unpool(z_develop, idx2))))
        x_bar = self.sig(self.conv_dec2(self.unpool(dec1, idx1)))
        
        return x_bar, z