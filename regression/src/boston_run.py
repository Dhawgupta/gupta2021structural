from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

boston = load_boston()
X,y   = (torch.tensor(boston.data).float(), torch.tensor(boston.target).float().reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
kfold = KFold(n_splits = 5)
train_index, valid_index = list(kfold.split(X_train, y_train))[0]
X_train2 = X_train[train_index]
y_train2 = y_train[train_index]


train_dataset = TensorDataset(X_train,y_train)
test_dataset = TensorDataset(X_test, y_test)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)
trainloader_all = torch.utils.data.DataLoader(train_dataset, batch_size= train_dataset.__len__(), shuffle = True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size= test_dataset.__len__(), shuffle = False)


print("dataset loaded")
input_dim = X_train.shape[1]
net = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
criterion = nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr = 5e-4)

num_epochs = 1000
train_losses = []
test_losses = []
for i in range(num_epochs):
    for x,y in trainloader_all:
        yhat = net(x.float())
        loss = criterion(yhat, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    # evaluate on test set

    with torch.no_grad():
        for x, y, in trainloader_all:
            yhat = net(x.float())
            train_loss = criterion(yhat, y)

        for x, y, in testloader:
            yhat = net(x.float())
            test_loss = criterion(yhat, y)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
    print(f"Epoch : {i}, Train Loss : {train_loss.item()}, Test Loss : {test_loss.item()}")

plt.ylim([0,100])
plt.plot( train_losses, label = 'train')
plt.plot( test_losses, label = 'test')
plt.legend()
plt.show()