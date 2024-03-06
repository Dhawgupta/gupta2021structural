class BaseProblemClassification(object):
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.seed = params['seed']


class BaseProblemRegression(object):
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.seed = params['seed']

