
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

    def train(self):
        pass

    def get_prediction(self):
        pass
