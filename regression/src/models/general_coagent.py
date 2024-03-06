'''
Gaussian Nodes with fixed standard deviation
# This is for the on policy case
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal



class GeneralCoagent(nn.Module):
    '''
    The continous coagent 
    '''
    def __init__(self, params):
        super(GeneralCoagent, self).__init__()
        self.in_features = params['in_features']
        self.model_specs = params['model_specification']
        self.num_layers = self.model_specs['num_layers']
        self.activation = self.model_specs.get('activation', 'none')
        self.std  = torch.tensor(params.get('model_std', 0.5), requires_grad= False)
        self.units_layer = params['units_layer']
        self.batch_size = params['batch_size']
        self.layers = []

        # make the model
        for i in range(self.num_layers + 1):
            if i == 0:
                self.layers.append(nn.Linear(self.in_features, self.units_layer ))
            elif i == self.num_layers :
                self.layers.append(nn.Linear(self.units_layer , 1)) # single output for regressions
            else:
                self.layers.append(nn.Linear(self.units_layer, self.units_layer ))
            # register parameters
            for name, param in self.layers[i].named_parameters():
                self.register_parameter(name = f'{i}-{name}', param = param)
        self.coagent_states = []

    def get_activation(self, x):
        if self.activation == 'none':
            return x
        if self.activation == 'relu':
            return f.relu(x)
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        if self.activation == 'tanh':
            return f.tanh(x)
        else:
            raise NotImplementedError

    def get_forward_mean(self, model_idx , x):
        return self.layers[model_idx](x)

    def sample_state(self, mean, std, greedy = False):
        if greedy :
            preaction = mean
        else:
            dist = Normal(mean, std)
            preaction = dist.sample()
        # apply the activation on action
        action = self.get_activation(preaction)
        return action, preaction

    def sample_all_coagent_states(self, x, greedy = False):
        # sample all the sataets for coagent (a forward inferencing step).
        states = []
        preactivations = []
        state = x
        for i in range(self.num_coagent_layers()):
            mean = self.get_forward_mean(i, state) #self.layers[i](state)
            state , preaction = self.sample_state(mean, self.std, greedy = greedy)
            states.append(state)
            preactivations.append(preaction)
        return states, preactivations

    def forward(self, x, greedy = False):
        with torch.no_grad():
            states, preactivations = self.sample_all_coagent_states(x, greedy = greedy)

        return self.layers[-1](states[-1]) , states,  preactivations

    def num_coagent_layers(self):
        return self.num_layers

