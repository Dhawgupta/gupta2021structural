
import torch.nn as nn
class ClassificationAgent(object):
    def __init__(self, params):
        '''
        Some default keys in the param dicionary
        in_features
        actions
        gamma
        alpha
        device
        '''
        self.alpha = params['alpha']
        self.in_features = params['in_features']
        self.num_classes = params['num_classes']
        self.device = params['device']
        self.updates = 0
        self.params = params
        self.optim_type = params['optimizer']


    def train(self, batch_x, batch_y):
        pass

    def get_prediction(self):
        pass


class RegressionAgent(object):
    def __init__(self, params):
        self.alpha = params['alpha']
        self.in_features = params['in_features']
        self.device = params['device']
        self.updates = 0
        self.params = params
        self.optim_type = params['optimizer']
        if self.params['agent'] == 'backprop':
            if self.optim_type == 'rmsprop':
                self.RMSpropBeta = params['RMSpropBeta']
            elif self.optim_type == 'adam':
                self.AdamBeta1 = params['AdamBeta1']
                self.AdamBeta2 = params['AdamBeta2']


    def evaluate(self, X, y):
        '''
        Evaluate the agent based on the given dataset
        Return : A singele loss value (could be MSE loss etc)
        '''
        raise NotImplementedError

    def train(self):
        pass

    def get_prediction(self):
        pass

    def get_objective(self):
        # return the specific objecitve / criteria
        # if self.objective_type == 'l1':
        #     return nn.L1Loss
        # if self.objective_type == 'l2':
        #     return nn.MSELoss
        # else:
        #     return NotImplementedError
        # not using L1Loss for now
        return nn.MSELoss