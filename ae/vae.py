import torch
from torch import nn
from torch.optim import Adam

class VAE(nn.Module):
    def __init__(self, in_shape, n_z):
        super().__init__()
        c, h, w = in_shape
        
        self.encoder = nn.Sequential(
            nn.Linear(c*h*w, 500), 
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),
            nn.Linear(500, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
        )
        
        self.z_mean = nn.Linear(1000, n_z)
        self.z_var = nn.Linear(1000, n_z)
        self.z_develop = nn.Linear(n_z, 1000)
        
        self.decoder = nn.Sequential(
            nn.Linear(1000, 500), 
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, c*h*w), 
            nn.Sigmoid()
        )

    def sample_z(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        eps = torch.randn(std.size()).to(device)
        return (eps * std) + mean

    def encode(self, x):
        x = self.encoder(x)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        out = self.z_develop(z)
        out = self.decoder(out)
        return out

    def forward(self, x):
        x_shape = x.shape
        x = x.view(x.shape[0], -1)
        
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        x_bar = self.decode(z)
        
        x_bar = x_bar.reshape(x_shape)
        return x_bar, z, mean, logvar
