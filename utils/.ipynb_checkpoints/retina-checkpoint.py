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

def make_dataset():
    path_to_files = "./data/Glaucoma"
    array_of_images = []
    
    for _, file in enumerate(os.listdir(path_to_files)): # to check if file has a certain name   
        file = path_to_files + "/" + file
        image = cv2.imread(file)
        single_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        single_array = cv2.resize(single_array, (128, 128), interpolation = cv2.INTER_AREA)
        array_of_images.append(single_array)

    x_train = array_of_images
    y_train = list(np.zeros(len(x_train), dtype=int))

    path_to_files = "./data/Non-Glaucoma"    
    array_of_images1 = []

    for _, file in enumerate(os.listdir(path_to_files)): # to check if file has a certain name   
        file = path_to_files + "/" + file
        image = cv2.imread(file)
        single_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        single_array = cv2.resize(single_array, (128, 128), interpolation = cv2.INTER_AREA)
        array_of_images1.append(single_array)

    x_train1 = array_of_images1
    y_train1 = list(np.ones(len(x_train1), dtype=int))
    
    x = np.array(x_train + x_train1)
    y = np.array(y_train + y_train1)
    x, y = shuffle(x, y, random_state=5)

    np.savez_compressed('retina_rgb.npz', x_train=x, y_train=y)
################################################################################
################################################################################
################################################################################

def load_retina(path='./data/retina_rgb.npz'):
    f = np.load(path)
    x, y = f['x_train'], f['y_train']
    f.close()
    
    x = x.astype(np.float32)
    x = np.transpose(x.data, (0, 3, 1, 2))
    x = np.divide(x, 255.)
    y = y.astype(np.int32)
    print('Retina samples', x.shape, y.shape)
    return x, y
    
class RetinaDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_retina()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))