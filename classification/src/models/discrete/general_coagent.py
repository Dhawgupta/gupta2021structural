'''
This type of model will use only linear coagents in all their layers i.e. all nodes in tehg graph are just stochastic nodes
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.categorical import Categorical


# returns a model


class GeneralCoagent(nn.Module):
    def __init__(self, params):
        super(GeneralCoagent, self).__init__()
        '''
         "model_specification" : {"num_layers" : 1 },
         "num_coagents" : [4, 8, 16, 32, 64, 128]
        '''
        self.in_features = params['in_features']
        self.num_classes = params['num_classes']
        self.model_specs = params['model_specification']
        self.num_layers = self.model_specs['num_layers']
        self.num_coagents = params['units_layer']
        self.batch_size = params['batch_size']
        self.layers = [] # this will contain the model for different layers eg. single layered nn, makes this a double element list element list

        for i in range(self.num_layers + 1):
            if i == 0:
                self.layers.append(nn.Linear(self.in_features[0], self.num_coagents *2 ))

            elif i == self.num_layers :
                self.layers.append(nn.Linear(self.num_coagents , self.num_classes))
            else:
                self.layers.append(nn.Linear(self.num_coagents, self.num_coagents * 2))

            for name, param in self.layers[i].named_parameters():
                self.register_parameter(name = f'{i}-{name}', param = param)
            # self.register_parameter(name=f'{i}', param=self.layers[i].parameters())

        self.coagent_states = [] # this will contain a list of states for all stocahastic nodes, len = self.num_layers
        self.all_softmax = []

    def get_forward_softmax( self, model , x):
        return f.softmax(model(x).reshape([-1, self.num_coagents, 2]), dim = 2)

    def sample_state(self, softmax_values):
        # batch_data = torch.zeros([softmax_values.shape[0], self.num_coagents])
        # for i, b in enumerate(softmax_values):
        #     d = torch.multinomial(b, 1)
        #     batch_data[i] = d.reshape([-1])
        #
        dist = Categorical(softmax_values)
        batch_data = dist.sample().float()
        return batch_data
        # make it faster using

    def sample_all_coagent_states(self, x, greedy = False):
        # sample a full bactch of coiagent states
        states = []
        state = x
        self.all_softmax = []
        for i in range(self.num_layers):
            state_softmax_probs = self.get_forward_softmax(self.layers[i], state)
            self.all_softmax.append(state_softmax_probs)
            if greedy :
                state = state_softmax_probs.max(dim = 2)[1].float()
            else:
                state = self.sample_state(state_softmax_probs)
            states.append(state)
        # print(greedy)
        return states

    def forward(self, x, greedy = False):
        # sample a state from coagent
        with torch.no_grad():
            states = self.sample_all_coagent_states(x, greedy = greedy)

        return self.layers[-1](states[-1]) , states # removed the softmax for now

    def num_coagent_layers(self):
        return self.num_layers

