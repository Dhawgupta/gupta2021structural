import torch
import torchvision
from torch.utils.data import TensorDataset
# import os, sys
# sys.path.append(os.getcwd())

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from src.problems.tunable_correlation_difficulty import load_correlated
from src.problems.base_problem import BaseProblemRegression
import matplotlib.pyplot as plt

class Correlated(BaseProblemRegression):
    def __init__(self, params):
        super(Correlated, self).__init__(params)

        self.difficulty = params['difficulty']
        self.test_train_ratio = 100
        self.bound =1
        self.num_means = params['means']
        self.segment_length = params['segments']

        # X, y = (torch.tensor(boston.data).float(), torch.tensor(boston.target).float().reshape(-1, 1))
        self.X_train = torch.from_numpy(load_correlated.get_train(num_means=self.num_means, segment_length=self.segment_length, difficulty=self.difficulty, bound=self.bound, batch_size=self.batch_size))
        self.y_train = load_correlated.target_function(self.X_train)

        self.X_valid = load_correlated.get_test(test_set_size=self.X_train.shape[0]//self.test_train_ratio, bound=self.bound)
        self.y_valid = load_correlated.target_function(self.X_valid )
        
        self.X_test = load_correlated.get_test(test_set_size=self.X_train.shape[0]//self.test_train_ratio, bound=self.bound)
        self.y_test = load_correlated.target_function(self.X_test )
        # self.X_trainfull, self.X_test, self.y_trainfull, self.y_test = train_test_split(X, y, test_size=0.1, random_state = 50 , shuffle = True)
        # p1 = 0.1 / (0.9)
        # self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_trainfull, self.y_trainfull, test_size = p1, random_state= 84, shuffle = True  )

        self.in_features = self.X_train.shape[1] # get the input dim
        self.X_train = torch.tensor(self.X_train).float()
        self.X_valid = torch.tensor(self.X_valid).float()
        self.X_test = torch.tensor(self.X_test).float()
        self.y_train = torch.tensor(self.y_train).float().reshape(-1, 1)
        self.y_valid = torch.tensor(self.y_valid).float().reshape(-1, 1)
        self.y_test = torch.tensor(self.y_test).float().reshape(-1, 1)



    def getTrainSet(self):
        return self.X_train, self.y_train

    def getTestSet(self):
        return self.X_test, self.y_test

    def getValidSet(self):
        return self.X_valid, self.y_valid

    def applyTransform(self, dataset):
        '''
        No transformation for this case
        '''
        return dataset

    def get_feature_size(self):
        return self.in_features

    def getBatchLoader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    def getFullLoader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=dataset.tensors[0].shape[0], shuffle=False)

if __name__ == '__main__':
    params  = {
        'batch_size' : 32,
        'seed' : 0
    }
    d = Correlated(params)
    # print("Dataset loaded")
    # y1 = d.y_train.reshape(-1).numpy()
    # y1.sort()
    # y2 = d.y_test.reshape(-1).numpy()
    # y3 = d.y_valid.reshape(-1).numpy()
    # y2.sort()
    # y3.sort()
    # plt.hist(y1, label='train')
    # plt.hist(y2, label='test')
    # plt.hist(y3, label='valid')
    # plt.legend()
    # plt.show()