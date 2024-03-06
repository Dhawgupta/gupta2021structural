import torch

class BaseProblemClassification(object):
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.seed = params['seed']


class BaseProblemRegression(object):
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.seed = params['seed']

    def getBatchLoader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def getFullLoader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)


