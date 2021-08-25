from PIL import Image
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
################################################################################
################################################################################
################################################################################

def load_brain(path='./data/brain_rgb.npz'):
    f = np.load(path)
    x, y = f['x_train'], f['y_train']
    f.close()
    
    x = x.astype(np.float32)
    x = np.transpose(x.data, (0, 3, 1, 2))
    x = np.divide(x, 255.)
    y = y.astype(np.int32)
    print('Brain samples', x.shape, y.shape)
    return x, y
    
class BrainDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_brain()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))