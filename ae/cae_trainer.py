import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

#=========================================================================
#======================= Early Stopping ==================================
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
                
        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)
                
#=========================================================================
#================== AutoEncoder Trainer ==================================
                
class CAETrainer:
    def __init__(self, model, data, pretrain_path='saved_models/CAE/cae_mnist.pkl'):
        self.data = data
        self.model = model
        self.pretrain_path = pretrain_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def train(self, lr, epochs, batch_size, early_patience):
        es = EarlyStopping(patience=early_patience)
        dataloader = DataLoader(dataset=self.data, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        time_elapsed = 0
        total_loss = []
        since = time.time()
        
        for epoch in range(epochs):
            total_loss.append(0)
            
            for batch_idx, (x, _, _) in enumerate(dataloader):
                x = x.to(self.device)
                x_bar, _ = self.model(x)
                loss = criterion(x_bar, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss[-1] += loss.item()
            
            time_elapsed = time.time() - since
            total_loss[-1] /= (batch_idx + 1)
            print("epoch {}/{}, loss: {:.4f}, time: {:.0f}m {:.0f}s".format(
                epoch, epochs, total_loss[-1], time_elapsed // 60, time_elapsed % 60))
            
            if es.step(total_loss[-1]):
                print("Training stoped with early stopping")
                break
        torch.save(self.model.state_dict(), self.pretrain_path)
        print("model saved to {}.".format(self.pretrain_path))
        
        df = pd.DataFrame({
            "Epoch": np.arange(epoch + 1),
            "Recon Loss": total_loss})
        df.to_csv(self.pretrain_path.split('.')[0] + "_loss.csv")
        
        return df
            