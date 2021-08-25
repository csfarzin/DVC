import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

class CVAE(nn.Module):
    def __init__(self, in_shape, n_z):
        super().__init__()
        self.in_shape = in_shape
        self.n_z = n_z
        c, h, w = in_shape
        self.z_dim = h//2**2 # receptive field downsampled 2 times
        
        self.encoder = nn.Sequential(
            #nn.BatchNorm2d(c),
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
            # CIFAR 32, 16, 16 # Mnist 32, 14, 14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            # CIFAR 64, 8, 8 # Mnist 64, 7, 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.z_mean = nn.Linear(64 * self.z_dim**2, n_z)
        self.z_var = nn.Linear(64 * self.z_dim**2, n_z)
        self.z_develop = nn.Linear(n_z, 64 * self.z_dim**2)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def sample_z(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        eps = torch.randn(std.size()).to(device)
        return (eps * std) + mean
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        out = self.z_develop(z)
        out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
        out = self.decoder(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, z, mean, logvar
