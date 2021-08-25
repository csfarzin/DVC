import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def load_cifar10(path='./data'):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(root=path, train=True, download=False, transform=transform)
    test = datasets.CIFAR10(root=path, train=False, download=False, transform=transform)
    
    train.data = np.concatenate((train.data, test.data)).astype(np.float32)
    train.targets = np.concatenate((train.targets, test.targets)).astype(np.int32)
    
    x = train.data
    x = np.transpose(x, (0, 3, 1, 2))
    x = np.divide(x, 255.)
    y = np.array(train.targets)
    print('CIFAR10 samples', x.shape)
    return x, y

def load_cifar100(path='./data'):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR100(root=path, train=True, download=False, transform=transform)
    test = datasets.CIFAR100(root=path, train=False, download=False, transform=transform)
    
    train.data = np.concatenate((train.data, test.data))
    train.targets = np.concatenate((train.targets, test.targets))
    
    x = np.transpose(train.data, (0, 3, 1, 2))
    x = np.divide(x, 255.)
    y = train.targets
    print('CIFAR100 samples', x.shape)
    return x, y


class CIFAR10Dataset(Dataset):

    def __init__(self):
        self.x, self.y = load_cifar10()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx]), torch.tensor(np.array(idx))
    
class CIFAR100Dataset(Dataset):

    def __init__(self):
        self.x, self.y = load_cifar100()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx]), torch.tensor(np.array(idx))