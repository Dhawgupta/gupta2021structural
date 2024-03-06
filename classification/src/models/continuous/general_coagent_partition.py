'''
For the partitioning experiment
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

# returns a model
EPS = 1e-3

# returns the activation funciton class
def get_activation( activation):
    if activation == 'relu':
        return nn.ReLU()
    if activation == 'sigmoid' :
        return nn.Sigmoid()
    if activation == 'tanh':
        return nn.Tanh()

class GeneralCoagentPartition(nn.Module):
    '''
    The continous coagent
    '''
    def __init__(self, params):
        super(GeneralCoagentPartition, self).__init__()
        self.in_features = params['in_features']
        self.num_classes = params['num_classes']
        self.model_specs = params['model_specification']
        self.num_layers = self.model_specs['num_layers']
        self.activation = self.model_specs.get('activation', 'none')
        self.std  = torch.tensor(params.get('model_std', 0.5), requires_grad= False)
        # for this experiment we are keeping the penulitimate layer as stochstic
        self.model_partition = params.get('model_partition','d' * (self.num_layers -1) + 's')
        print(f"Model partition : {self.model_partition}")
        self.num_coagents = params['units_layer']
        self.batch_size = params['batch_size']

        self.layers = []
        self.layers.append(nn.Sequential())
        count = 0
        self.actual_layers = self.model_partition.count('s')  # the actual number of stochastic layers
        # make the model
        for i, l in enumerate(self.model_partition):
            if l == 's':  # then split the layer
                if i == 0:
                    self.layers[-1].add_module(f"layer-{count}-weight",
                                               nn.Linear(self.in_features[0], self.num_coagents ))
                else:
                    self.layers[-1].add_module(f"layer-{count}-weight",
                                               nn.Linear(self.num_coagents, self.num_coagents ))
                    # switch to a different model now
                self.layers.append(nn.Sequential())
            if l == 'd':  # deterministic layer
                # add activation
                if i == 0:
                    self.layers[-1].add_module(f"layer-{count}-weight",
                                               nn.Linear(self.in_features[0], self.num_coagents))
                else:
                    self.layers[-1].add_module(f"layer-{count}-weight",
                                               nn.Linear(self.num_coagents, self.num_coagents))
                self.layers[-1].add_module(f"layer-{count}-activation", get_activation(self.activation))
            count += 1

        self.layers[-1].add_module(f"layer-{count}-weight", nn.Linear(self.num_coagents, self.num_classes))

        for l in self.layers:
            for name, param in l.named_parameters():
                # print(name, param)
                self.register_parameter(name=f'{name.replace(".", "-")}', param=param)
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
        for i in range(self.actual_layers):
            mean = self.get_forward_mean(i, state)
            state , preaction = self.sample_state(mean, self.std, greedy = greedy)
            states.append(state)
            preactivations.append(preaction)
        return states, preactivations

    def forward(self, x, greedy = False):
        with torch.no_grad():
            states, preactivations = self.sample_all_coagent_states(x, greedy = greedy)

        if len(states) == 0:
            return self.layers[-1](x), states, preactivations
        else:
            return self.layers[-1](states[-1]) , states, preactivations


    def num_coagent_layers(self):
        return self.actual_layers