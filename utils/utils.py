import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment as linear_assignment

def load_mnist(path='./data'):    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
    train = datasets.MNIST(root=path, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=path, train=False, download=True, transform=transform)
    
    train.data = np.concatenate((train.data, test.data)).astype(np.float32)
    train.targets = np.concatenate((train.targets, test.targets)).astype(np.int32)
    
    x = train.data
    x = np.expand_dims(x, 1)
    x = np.divide(x, 255.)
    #x = np.concatenate((x, x, x), axis=1)
    y = train.targets
    print('MNIST samples', x.shape)
    return x, y


class MnistDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_mnist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
