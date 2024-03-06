'''
Gaussian Nodes with fixed standard deviation
# This is for the on policy case
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

# returns a model
EPS = 1e-3


class CritcLayer(nn.Module):
    '''
    The continous coagent
    '''
    def __init__(self, num_coagents, batch_size, features):
        super(CritcLayer, self).__init__()
        self.in_features = features
        self.num_coagents = num_coagents
        self.batch_size = batch_size

        self.networks = []

        # make the model
        for i in range(self.num_coagents ):
            self.networks.append(nn.Linear(self.in_features, 1))
            self.networks[-1].weight.data.fill_(0)
            self.networks[-1].bias.data.fill_(0)
            for name , param in self.networks[-1].named_parameters():
                self.register_parameter(name = f'{i}-{name}', param = param)


    def forward(self, x):
        outputs = []
        for i, data in enumerate(x):
            output = self.networks[i](data)
            outputs.append(output.unsqueeze(0))
        # returns the batch output for each coagent
        return torch.cat(outputs).reshape(self.num_coagents, -1).permute(1, 0)


