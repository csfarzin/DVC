from PIL import Image
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset

def make_dataset():
    path_to_files = "./data/ISIC2018/ISIC2018_Task3_Training_Input"
    array_of_images = []
    
    for _, file in enumerate(os.listdir(path_to_files)): # to check if file has a certain name   
        fname = path_to_files + "/" + file
        
        if fname.endswith('.jpg'):
            image = cv2.imread(fname)
            single_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            single_array = cv2.resize(single_array, (128, 128), interpolation = cv2.INTER_AREA)
            array_of_images.append(single_array)

    x_train = array_of_images

###############################################################################
#### reading class types ######################################################
    df = pd.read_csv(
        'data/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
    
    y = np.zeros(len(df.index))
    i = 0
    for col in df.columns[1:8]:
        idx = df[col].to_numpy().astype(bool)
        y[idx] = i
        i += 1
    
    x = np.array(x_train)
    x, y = shuffle(x, y, random_state=5)

    np.savez_compressed('skin_rgb.npz', x_train=x, y_train=y)

################################################################################
################################################################################
################################################################################

def load_skin(path='./data/skin_rgb.npz'):
    f = np.load(path)
    x, y = f['x_train'], f['y_train']
    f.close()
    
    x = x.astype(np.float32)
    x = np.transpose(x.data, (0, 3, 1, 2))
    x = np.divide(x, 255.)
    y = y.astype(np.int32)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.85, random_state=5)
    print('Skin samples shapes x_train: {}, x_test: {}'.format(x_train.shape, x_test.shape))
    
    return x_train, y_train, x_test, y_test
    
class SkinDatasetTrain(Dataset):

    def __init__(self):
        self.x, self.y, _, _ = load_skin()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(
            self.x[idx]), torch.tensor(self.y[idx]), torch.tensor(idx)
    
class SkinDatasetTest(Dataset):

    def __init__(self):
        _, _, self.x, self.y = load_skin()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(
            self.x[idx]), torch.tensor(self.y[idx]), torch.tensor(idx)