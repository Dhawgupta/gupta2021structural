import torch
import torch.nn as nn



def get_activation( activation):
    if activation == 'relu':
        return nn.ReLU()
    if activation == 'sigmoid' :
        return nn.Sigmoid()
    if activation == 'tanh':
        return nn.Tanh()

def make_nn_layers(model, in_features, out, num_layers, activation , num_nodes_layer = None):

    if num_layers == 0:  # linear classifier
        model.add_module("0", nn.Linear(in_features, out_features=out))
    else:
        model.add_module("0", nn.Linear(in_features, num_nodes_layer))
        model.add_module("0-a",  get_activation(activation))
        for i in range(num_layers - 1):
            model.add_module(f"{i + 1}", nn.Linear(num_nodes_layer, num_nodes_layer))
            model.add_module((f"{i + 1}-a"), get_activation(activation))

        model.add_module(f"{num_layers}", nn.Linear(num_nodes_layer, out))
    return model


class GeneralNeuralNet(nn.Module):
    def __init__(self, params):
        super(GeneralNeuralNet, self).__init__()
        self.in_features = params['in_features']
        self.num_outputs = 1
        self.model_specs = params['model_specification']
        self.model = nn.Sequential()
        self.num_layers = self.model_specs["num_layers"]
        self.activation = self.model_specs.get('activation', "relu")
        self.num_nodes_layer = None
        if self.num_layers > 0 :
            self.num_nodes_layer = params['units_layer']
        make_nn_layers(self.model, self.in_features, self.num_outputs, self.num_layers, self.activation, self.num_nodes_layer)

    def forward(self, x):
        return self.model(x)