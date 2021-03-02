# from torchvision.datasets import CIFAR10
#
# trainset = CIFAR10('/Users/IrmavandenBrandt/Downloads/test_data_torchvision', train=True, transform=None, download=True)
# testset = CIFAR10('/Users/IrmavandenBrandt/Downloads/test_data_torchvision', train=False, transform=None)

import pickle

def load_embedding(filename):
    with open(filename, 'rb') as f:
        e = pickle.load(f)
    return e

#%%
embedding = load_embedding('/Users/IrmavandenBrandt/Downloads/embedding_pcam-small.p')

#%%
embedding_isic = load_embedding('/Users/IrmavandenBrandt/Downloads/embedding_isic2018.p')
#%%
embedding_stl10 = load_embedding('/Users/IrmavandenBrandt/Downloads/embedding.p')
#%%
embedding_stl10_resnet50 = load_embedding('/Users/IrmavandenBrandt/Downloads/embedding_resnet50.p')
