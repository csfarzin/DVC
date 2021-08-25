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
        self.z_dim = h//2**4 # receptive field downsampled 2 times
        
        self.C_En1 = nn.Conv2d(c, 6, kernel_size=5, stride=1, padding=2)
        self.B_En1 = nn.BatchNorm2d(6)
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.relu = nn.ReLU()
        # 6, 64, 64
        
        self.C_En2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.B_En2 = nn.BatchNorm2d(16)
        # nn.MaxPool2d(2, 2, return_indices=True)
        # 16, 32, 32
        
        self.C_En3 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.B_En3 = nn.BatchNorm2d(32)
        # nn.MaxPool2d(2, 2, return_indices=True)
        # 32, 16, 16
        
        self.C_En4 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.B_En4 = nn.BatchNorm2d(64)
        # nn.MaxPool2d(2, 2, return_indices=True)
        # 64, 8, 8
        
        self.z_mean = nn.Linear(64 * self.z_dim**2, n_z)
        self.z_var = nn.Linear(64 * self.z_dim**2, n_z)
        self.z_develop = nn.Linear(n_z, 64 * self.z_dim**2)
        
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.CT_De4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.B_De4 = nn.BatchNorm2d(32)
        
        self.CT_De3 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=2)
        self.B_De3 = nn.BatchNorm2d(16)
        
        self.CT_De2 = nn.ConvTranspose2d(16, 6, kernel_size=5, stride=1, padding=2)
        self.B_De2 = nn.BatchNorm2d(6)
        
        self.CT_De1 = nn.ConvTranspose2d(6, c, kernel_size=5, stride=1, padding=2)
        self.sig = nn.Sigmoid()

    def sample_z(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        eps = torch.randn(std.size()).to(device)
        return (eps * std) + mean
        #std = logvar.mul(0.5).exp_()    
        #eps = torch.empty_like(std).normal_()
        #return eps.mul(std).add_(mean)
    
    def encoder(self, x):
        x, self.idx1 = self.pool(self.relu(self.B_En1(self.C_En1(x))))
        x, self.idx2 = self.pool(self.relu(self.B_En2(self.C_En2(x))))
        x, self.idx3 = self.pool(self.relu(self.B_En3(self.C_En3(x))))
        x, self.idx4 = self.pool(self.relu(self.B_En4(self.C_En4(x))))
        return x
    
    def decoder(self, x):
        x = self.relu(self.B_De4(self.CT_De4(self.unpool(x, self.idx4))))
        x = self.relu(self.B_De3(self.CT_De3(self.unpool(x, self.idx3))))
        x = self.relu(self.B_De2(self.CT_De2(self.unpool(x, self.idx2))))
        x = self.sig(self.CT_De1(self.unpool(x, self.idx1)))
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        out = self.z_develop(z)
        out = out.reshape(z.shape[0], 64, self.z_dim, self.z_dim)
        out = self.decoder(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, z, mean, logvar
