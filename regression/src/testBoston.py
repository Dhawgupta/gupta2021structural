from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

seed = 3
np.random.seed(seed)
torch.manual_seed(seed)

boston = load_boston()
X,y   = (torch.tensor(boston.data).float(), torch.tensor(boston.target).float().reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print(X_test.shape)
print(y_test.shape)
p1 = 0.1 / (0.9)
X_t, X_v, y_t,  y_v = train_test_split(X_train, y_train, test_size= p1, random_state= 0 )
print(X_v.shape)
print(y_v.shape)
