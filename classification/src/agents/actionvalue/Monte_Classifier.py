import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer
from src.agents.agent_template import ClassificationAgent


def make_nn_layers(model, in_features, out, num_layers, num_nodes_layer = None):
    if num_layers == 0:  # linear classifier
        model.add_module("0", nn.Linear(in_features[0], out_features=out))
    else:
        model.add_module("0", nn.Linear(in_features[0], num_nodes_layer))
        model.add_module("0-a", nn.ReLU())
        for i in range(num_layers - 1):
            model.add_module(f"{i + 1}", nn.Linear(num_nodes_layer, num_nodes_layer))
            model.add_module((f"{i + 1}-a"), nn.ReLU())
        model.add_module(f"{num_layers}", nn.Linear(num_nodes_layer, out))
    return model

class Monte_Classifier(ClassificationAgent):
    '''
    '''
    def __init__(self, params):
        super(Monte_Classifier, self).__init__(params)
        # number of coagents
        self.units_layer = params['units_layer']
        #self.nettype = params['model']
        self.batch_size = params['batch_size']
        self.in_features = params['in_features']
        self.num_classes = params['num_classes']
        self.model_specs = params['model_specification']
        self.num_layers = self.model_specs['num_layers']
        self.num_nn_layers = 0 # #hidden layers 0, as we are always using a linear layer
        self.num_outputs = params['num_outputs']

        self.determined = params['eval_greedy']
        self.running_loss = None
        self.coagent_layers = [] # will be a list of list of coagents
        self.coagent_layers_outs=[None]*self.num_layers # initialise output of each coagent layer 

        self.epsilon = params['epsilon']
        self.decrement = True if self.epsilon == 1 else False
        self.eps_dec = 1e-4
        self.eps_min = 0.01    
        #self.tau = params['tau']
        self.exploration = params['exploration']
        self.eps_min_reached = False

        self.activations = np.zeros((self.batch_size,self.units_layer))
        self.activations = np.expand_dims(self.activations, 0)
        self.optimizers = [] 
        self.criterion = nn.CrossEntropyLoss() 
        self.coagent_criterion = nn.MSELoss()
        
        # set up lists of coagent layers and their optimizers
        for i in range(self.num_layers):
            if i==0:
                self.coagent_layers = [nn.Linear(self.in_features[0], self.units_layer * self.num_outputs )]
                self.optimizers = [get_optimizer(self.optim_type)(self.coagent_layers[-1].parameters(), lr = self.alpha)]
            else:
                self.coagent_layers.append(nn.Linear(self.units_layer, self.units_layer * self.num_outputs ))
                self.optimizers.append( get_optimizer(self.optim_type)(self.coagent_layers[-1].parameters(), lr = self.alpha) )

        self.model = Policy(params)
        self.optimizer_model = get_optimizer(self.optim_type)(self.model.parameters(), lr=self.alpha)

    # for experiments with epsilon tied to the reward
    def decrement_epsilon_tiedtorewards(self, delta):
        if self.epsilon == self.eps_min:
            self.eps_min_reached = True
        if not self.eps_min_reached:
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        else:
            self.epsilon = -delta.sum() / 1000

    def decrement_epsilon(self):       
        return self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def boltzmann(self,q_values):
        numerator = torch.exp(q_values/self.tau)
        denominator = torch.sum(torch.exp(q_values/self.tau), 0)
        return (numerator/denominator)

    def coagent_forward(self, state, deterministic_runs=False):
        """ 
        sample from state based on epsilon-greedy policy for each coagent.
        """
        self.current_q=[]
        for coagent_layer in self.coagent_layers:
            
            q_values = coagent_layer(state)  
            q_values = q_values.reshape([-1, self.units_layer, self.num_outputs])
            max_actions = torch.argmax(q_values, dim = 2)  #(batch_size, units_layer)

            if deterministic_runs:
                state = max_actions.numpy()
            else:
                random_action_array = np.random.randint(0, self.num_outputs, (state.shape[0], self.units_layer)) #(batch_size, units_layer)
                #2d_array_greedy_epsilons of size bathc_size x units_layer
                random_probs = np.random.random((state.shape[0], self.units_layer)) #(batch_size, units_layer)
                state= np.where(random_probs>self.epsilon, max_actions, random_action_array)

            # get the q-values according to what action has been taken
            self.current_q.append(q_values.gather(2, torch.from_numpy(state).unsqueeze(2) ).squeeze())
            state = torch.from_numpy(state).float()

        # main agent
        q_values_final = self.model(state) # shape: (num of images in batch, num_classes)
        return q_values_final

    def get_prediction_class_probs(self, x, deterministic_runs=False):
        with torch.no_grad():
            class_probs = self.coagent_forward(x, deterministic_runs)
            return class_probs

    def get_predicition_class(self, x, deterministic_runs=False):
        with torch.no_grad():
            return torch.max(self.coagent_forward(x, deterministic_runs).data, 1)[1]

    def train(self, batch_x, batch_y):
        """
        training loop for the coagents and the main agent
        """
        class_probs = self.coagent_forward(batch_x)
        loss = self.criterion(class_probs, batch_y) # this is the negative reward cross-entropy
        # update the main optimizer
        self.optimizer_model.zero_grad()
        loss.backward()
        self.optimizer_model.step()

        with torch.no_grad():
            delta_loss = nn.CrossEntropyLoss(reduce=False)
            delta = - delta_loss(class_probs, batch_y) # use the negative value of loss as reward
        
        for j, coagent_layer in enumerate(self.coagent_layers):

            self.optimizers[j].zero_grad()
            c_q_current = self.current_q[j]

            delta_repeated = delta.repeat(self.units_layer).reshape([self.units_layer,delta.shape[0] ]) # delta repeated along columns, shape: (self.batch_size,self.units_layer)
            delta_repeated = torch.transpose(delta_repeated,0,1)

            coagent_loss = self.coagent_criterion(c_q_current, delta_repeated) # MSELoss  shape: (self.batch_size,self.units_layer)

            coagent_loss.backward()
            self.optimizers[j].step()
        
        if self.running_loss is None:
            self.running_loss = loss.item()
        
        self.running_loss = self.running_loss * 0.99 + .01 * loss.item()
        if self.decrement:
            self.epsilon = self.decrement_epsilon()
        return loss.item()

class Policy(nn.Module):
    """
    The final classifier
    """
    def __init__(self, params ):
        super(Policy, self).__init__()

        self.num_classes = params['num_classes']
        self.model_specs = params['model_specification']
        self.units_layer = params['units_layer']
        self.model = nn.Sequential()
        self.num_layers = self.model_specs["num_layers"]

        self.num_nodes_layer = None
        self.num_nn_layers = 0

        make_nn_layers(self.model, [self.units_layer], self.num_classes, self.num_nn_layers, self.num_nodes_layer)

    def forward(self, x):
        q_values = self.model(x)   
        # return a list with the probability of each action over the action space
        
        return q_values
