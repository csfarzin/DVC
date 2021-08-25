# DVC
Deep Variational Clustering Framework for Self-labeling Large-scale Medical Images
This is the official PyTorch implementation of the DVC paper:
```
@Article{

}
```



## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

## Dataset
* MNIST
* Skin Cancer
* REFUGE-2

## Training
To Train DVC run cvae_idec.py
```
python cvae_idec.py --batch_size 256 --lr 0.001
optional arguments:
--lr                          Learnig rate
--n_clusters                  Number of cluster
--n_z                         Size of embbeding layer
--batch_size                  Number of images in each mini-batch [default value is 512]
--dataset-name                Name of the dataset (e.g., mnist, skin, retina)
--pretrain_path               Path of pretrained model (e.g., "saved_models/VAE/cvae_cifar10.pkl")
--early_patience              Number of epochs before triggering the early stopping.
--gamma                       Coefficient of clustering loss
--update_interval             Specify the update interval of target distribution
```


