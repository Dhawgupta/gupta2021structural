import torch.optim as optim

def get_optimizer(optim_name):
    '''
    Uses PyTorch inbuilt optimizers
    '''
    if optim_name == 'rmsprop':
        return optim.RMSprop
    if optim_name == 'adam':
        return optim.Adam
    if optim_name == 'sgd':
        return optim.SGD
    else:
        raise NotImplementedError