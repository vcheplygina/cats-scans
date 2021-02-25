from torchvision.datasets import CIFAR10

trainset = CIFAR10('/Users/IrmavandenBrandt/Downloads/test_data_torchvision', train=True, transform=None, download=True)
testset = CIFAR10('/Users/IrmavandenBrandt/Downloads/test_data_torchvision', train=False, transform=None)
# %%


# import pickle
# objects = []
# with (open("/Users/IrmavandenBrandt/PycharmProjects/cats-scans/src/task2vec_fullcode/outputs/2021-02-11/09-45-04"
#            "/embedding_test_stl10.p", "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break

import pandas as pd
import task2vec
import variational
from task_similarity import load_embedding
import pickle

# object = load_embedding("/Users/IrmavandenBrandt/PycharmProjects/cats-scans/src/task2vec_fullcode/outputs/2021-02-11/09-45-04"
#            "/embedding_test_stl10.p")

objects = []
with (open("/src/task2vec_fullcode/outputs/2021-02-11/09-45-04"
           "/embedding_test_stl10.p", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
print(objects[0])