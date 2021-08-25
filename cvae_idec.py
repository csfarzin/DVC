from __future__ import print_function, division

import gc
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("ggplot")

from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader

from plot.tsne import Tsne
from utils.skin import SkinDatasetTrain, SkinDatasetTest
from utils.retina import RetinaDataset
from utils.utils import MnistDataset, cluster_acc
from utils.cifar import CIFAR10Dataset, CIFAR100Dataset
from utils.brain import BrainDataset

from ae.vae_trainer import VAETrainer
#from ae.cvae_glau import CVAE
from ae.cvae_cifar import CVAE
#from ae.cvae import CVAE
from ae.vae import VAE


import warnings
warnings.filterwarnings('ignore')

import os
def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)

makedir('./saved_models/VAE')
makedir('./images')
makedir('./TSNE_images')

#def imshow(img, y):
#    classes = ('plane', 'car', 'bird', 'cat', 
#               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#    i = 0
#    for img_i, y_i in zip(img, y):
#        #npimg = img_i.reshape(32, 32, 3)
#        npimg = img_i.reshape(28, 28)
#        plt.imshow(npimg)
#        plt.title(classes[y_i])       
#        plt.savefig('image_' + ctime() + '.png')
        
#################################################################################
### Visaulizing Img and Img_bar ####################################################

# The next three methods to visualize input/output of our model side-by-side
def hstackimgs(min, max, images):
    return np.hstack([images[i] for i in range(min, max)])

def sqstackimgs(length, height, images):
    return np.vstack([hstackimgs(i*length, (i+1)*length, images) for i in range(height)])

def sbscompare(images1, images2, length, height):
    A = sqstackimgs(length, height, images1)
    B = sqstackimgs(length, height, images2)
    #C = np.ones((A.shape[0], 32, 3))
    #C = np.ones((A.shape[0], 28))
    return np.hstack((A, B))        

#################################################################################
#################################################################################
class IDEC(nn.Module):

    def __init__(self,
                 n_input,
                 n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path='saved_models/VAE/cvae_cifar10.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = CVAE(in_shape=n_input, n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
    ### Pretrain AE #################################################################
    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # load pretrain weights
        else:
            self.ae.load_state_dict(torch.load(self.pretrain_path))
            print('load pretrained ae from', path)

    def forward(self, x):
        x_bar, z, _, _ = self.ae(x)     
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q

#################################################################################
### Target Distribution #########################################################

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

#################################################################################
### Pretraining The AutoEncoder #################################################

def pretrain_ae(model):
    
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)
    
    trainer = VAETrainer(model, dataset, pretrain_path=args.pretrain_path)
    trainer.train(lr=args.lr, epochs=100, batch_size=args.batch_size, early_patience=20)

#################################################################################
### Fine Tuning #################################################################
def fw_model(model, loader):
    x_bar, z, _, _ = fw_ae(model, loader)

    q = 1.0 / (1.0 + torch.sum(
        torch.pow(
            z.unsqueeze(1).to(device) - model.cluster_layer, 2), 2) / model.alpha)
    q = q.pow((model.alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()

    return x_bar, q, z

def fw_ae(model, loader):
    x_bar, z, mean, logvar = [], [], [], []
    i = 0
    for x, _, _ in loader:
        x = x.to(device)
        o1, o2, o3, o4 = model.ae(x)
        x_bar.append(o1.detach().cpu())
        z.append(o2.detach().cpu())
        mean.append(o3.detach().cpu())
        logvar.append(o4.detach().cpu())
        
        #gc.collect()
        #torch.cuda.empty_cache()
        #r = torch.cuda.memory_reserved(0) 
        #a = torch.cuda.memory_allocated(0)
        #print('xxxxxxxxx:  ', r, a)
    x_bar = torch.cat(x_bar)
    z = torch.cat(z)
    mean = torch.cat(mean)
    logvar = torch.cat(logvar)
    
    return x_bar, z, mean, logvar


def train_idec(model):

    #model.pretrain(args.pretrain_path)
    model.pretrain()
    
    nmi = []
    acc = []
    ari = []
    recon_loss_t = []
    kl_loss_t = []
    loss_t = []
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # cluster parameter initiate
    #data = dataset.x
    #data = torch.tensor(data).to(device)
    y = dataset.y

    x_bar, z, _, _ = fw_ae(model, loader)
    #x_bar, z, _, _ = model.ae(data)
    
    z_trunc = z[:2000]
    y_trunc = y[:2000]
    
    plott = Tsne(2, z_trunc.numpy(), '0')
    plott.tsne_plt(y_trunc)
    
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.numpy())
    nmi_k = nmi_score(y_pred, y)
    print("nmi score={:.4f}".format(nmi_k))

    z = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    
    time_elapsed = 0
    since = time.time()
############################################################################
############################################################################
    epochs = 20
    model.train()
    for epoch in range(epochs):
        nmi.append(0)
        acc.append(0)
        ari.append(0)
        recon_loss_t.append(0)
        kl_loss_t.append(0)
        loss_t.append(0)
        
        if epoch % args.update_interval == 0:

            #_, tmp_q = model(data)
            _, tmp_q, z = fw_model(model, loader)
            
            if epoch == 5 or epoch == 10 or epoch == 15:
                z_trunc = z[:2000]
                plott = Tsne(2, z_trunc.numpy(), str(epoch))
                plott.tsne_plt(y_trunc)
            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            p = p.to(device)
            
            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            acc[-1] = cluster_acc(y, y_pred)
            nmi[-1] = nmi_score(y, y_pred)
            ari[-1] = ari_score(y, y_pred)
            
            time_elapsed = time.time() - since
            print('epoch {}/{}:'.format(epoch + 1, epochs), ' Acc {:.4f}'.format(acc[-1]),
                  ', nmi {:.4f}'.format(nmi[-1]), ', ari {:.4f}'.format(ari[-1]),
                 ', time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol', args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
                
        for batch_idx, (x, _, idx) in enumerate(train_loader):
            x = x.to(device)
            idx = idx.to(device)

            x_bar, q = model(x)

            #reconstr_loss = F.mse_loss(x_bar, x)
            reconstr_loss = F.binary_cross_entropy(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = args.gamma * kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            recon_loss_t[-1] += reconstr_loss.item()
            kl_loss_t[-1] += args.gamma * kl_loss.item()
            loss_t[-1] += loss.item()
            #gc.collect()
            #torch.cuda.empty_cache()
            
        recon_loss_t[-1] /= (batch_idx + 1)
        kl_loss_t[-1] /= (batch_idx + 1)
        loss_t[-1] /= (batch_idx + 1)
    df = pd.DataFrame({
        "Epoch": np.arange(epoch + 1),
        "nmi": nmi,
        "acc": acc,
        "ari": ari,
        "recon_loss": recon_loss_t,
        "kl_loss": kl_loss_t,
        "total_loss": loss_t
    })
    
    loss_path = args.pretrain_path.split(".")
    loss_path = loss_path[0] + "_model_history.csv"
    df.to_csv(loss_path)            

    print("Saving the final model")
    model_path = args.pretrain_path.split(".")
    model_path = model_path[0] + "_final_model." + model_path[1]
    torch.save(model.state_dict(), model_path)
                        
#################################################################################
### Main Loop ###################################################################

if __name__ == "__main__":
    #seed = 7
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--pretrain_path', type=str, default=
                        'saved_models/VAE/cvae_cifar10.pkl')
    parser.add_argument('--early_patience', type=int, default=10)
    parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    #device = torch.device('cpu')
    #if args.cuda:
    #    torch.cuda.manual_seed(seed)

    if args.dataset == 'mnist':
        args.pretrain_path = 'saved_models/VAE/cvae_mnist.pkl'
        args.n_clusters = 10
        dataset = MnistDataset()
        args.n_input = dataset.x[0].shape
        print(args)
    
    if args.dataset == 'cifar10':
        args.pretrain_path = 'saved_models/VAE/cvae_cifar10.pkl'
        args.n_clusters = 10
        dataset = CIFAR10Dataset()
        args.n_input = dataset.x[0].shape
        print(args)

    if args.dataset == 'cifar100':
        args.pretrain_path = 'saved_models/VAE/cvae_cifar100.pkl'
        args.n_clusters = 100
        dataset = CIFAR100Dataset()
        args.n_input = dataset.x[0].shape
        print(args)
        
    if args.dataset == 'retina':
        args.pretrain_path = 'saved_models/VAE/cvae_retina.pkl'
        args.n_clusters = 2
        args.n_z = 2
        args.batch_size = 4
        dataset = RetinaDataset()
        args.n_input = dataset.x[0].shape
        print(args)
        
    if args.dataset == 'brain':
        args.pretrain_path = 'saved_models/VAE/cvae_brain.pkl'
        args.n_clusters = 2
        args.n_z = 2
        args.batch_size = 16
        dataset = BrainDataset()
        args.n_input = dataset.x[0].shape
        print(args)
    
    if args.dataset == 'skin':
        args.pretrain_path = 'saved_models/VAE/cvae_skin.pkl'
        args.n_clusters = 7
        dataset = SkinDatasetTrain()
        dataset_test = SkinDatasetTest()
        args.n_input = dataset.x[0].shape
        print(args)

###################################################################    
###################################################################    
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
    
    model = IDEC(
    n_input=args.n_input,
    n_z=args.n_z,
    n_clusters=args.n_clusters,
    alpha=1.0,
    pretrain_path=args.pretrain_path).to(device)
    
    train_idec(model)
    
    data = dataset.x
    x = data
    y = dataset.y
 
    x_bar, z, _, _ = fw_ae(model, loader)

    x_bar = x_bar.numpy()
    z = z.numpy()
    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        reshaped_x = x.reshape(x.shape[0], 3, 32, 32)
        reshaped_x = np.transpose(reshaped_x, (0, 2, 3, 1))
        
        reshaped_x_bar = x_bar.reshape(x_bar.shape[0], 3, 32, 32)
        reshaped_x_bar = np.transpose(reshaped_x_bar, (0, 2, 3, 1))
        
        cifar_fig = plt.figure()
        plt.ion()
        plt.imshow(sbscompare(reshaped_x, reshaped_x_bar, 5, 5))
        plt.axis('off')
        plt.ioff()
        plt.savefig('images/CVAE_output_' + args.dataset + '.png', bbox_inches='tight')
        
    if args.dataset == 'mnist':
        reshaped_x = x.reshape(x.shape[0], 28, 28)
        reshaped_x_bar = x_bar.reshape(x_bar.shape[0], 28, 28)
        
        mnist_fig = plt.figure()
        plt.ion()
        plt.imshow(sbscompare(reshaped_x, reshaped_x_bar, 5, 5))
        plt.axis('off')
        plt.ioff()
        plt.savefig('images/CVAE_output_mnist.png', bbox_inches='tight')

    if args.dataset == 'retina' or args.dataset == 'skin' or args.dataset == 'brain':
        reshaped_x = x.reshape(x.shape[0], 3, 128, 128)
        reshaped_x = np.transpose(reshaped_x, (0, 2, 3, 1))
        
        reshaped_x_bar = x_bar.reshape(x_bar.shape[0], 3, 128, 128)
        reshaped_x_bar = np.transpose(reshaped_x_bar, (0, 2, 3, 1))
        
        retina_fig = plt.figure()
        plt.ion()
        plt.imshow(sbscompare(reshaped_x, reshaped_x_bar, 2, 2))
        plt.axis('off')
        plt.ioff()
        plt.savefig('images/CVAE_output_' + args.dataset + '.png', bbox_inches='tight')
        
    z = z[:10000]
    y = y[:10000]
    
    plott = Tsne(2, z, 'final')
    plott.tsne_plt(y)
    
    #_, q = model(data)
    #q = q.detach().cpu().numpy()
    
    #print(y[0])
    #print(q.shape)
    #print(q[0])
