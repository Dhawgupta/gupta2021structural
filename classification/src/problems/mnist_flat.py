import torch
import torchvision
from torch.utils.data import TensorDataset
from src.problems.base_problem import BaseProblemClassification



class MnistFlat(BaseProblemClassification):
    def __init__(self, params):
        super(MnistFlat, self).__init__(params)
        self.trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                     download=True)
        

        self.testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                    download=True)
        self.classes = ('0','1','2','3','4','5','6','7','8','9')
        self.in_features = [784]
        self.num_classes = len(self.classes)

    def getTrainLoader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                       shuffle=True)
    def getTrainSet(self):
        return self.trainset

    def getTestSet(self):
        return self.testset

    def applyTransform(self, dataset):
        dataset = dataset/255
        dataset = ( dataset-0.5 ) /0.5
        dataset  = dataset.reshape(dataset.shape[0],dataset.shape[1]*dataset.shape[2])
        return dataset

    def getTestLoader(self, dataset):
        # FIXME change the batch size
        # TODO try this out
        return torch.utils.data.DataLoader(dataset, batch_size= dataset.__len__(),
                                                      shuffle=False)

    def get_classes(self):
        return self.classes

    def get_feature_size(self):
        return self.in_features

    def get_num_classes(self):
        return self.num_classes


if __name__ == '__main__':
    params = {'batch_size': 4, "seed" : 1}
    problem = MnistFlat(params)
    print("Loaded")

