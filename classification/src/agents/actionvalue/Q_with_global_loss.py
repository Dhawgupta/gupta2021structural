import numpy as np
import copy
from src.agents.agent_template import ClassificationAgent
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer

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

class Q_with_global_loss(ClassificationAgent):
    '''
    '''
    def __init__(self, params):
        super(Q_with_global_loss, self).__init__(params)
        # number of coagents
        self.num_coagents = params['units_layer']
        #self.nettype = params['model']
 
        self.in_features = params['in_features']
        self.num_classes = params['num_classes']
        self.model_specs = params['model_specification']
        self.num_layers = self.model_specs['num_layers']
        self.num_nn_layers = 0 # #hidden layers 0, as we are always using a linear layer
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
        self.num_outputs = params['num_outputs']


        #self.coagents = []
        self.optimizers = [] 
        self.criterion = nn.CrossEntropyLoss() 

        for j in range(self.num_layers):
            temp_coagent_list = []
            for i in range(self.num_coagents):
                temp_coagent_list.append(Coagent(params,j))
                self.optimizers.append(get_optimizer(self.optim_type)(temp_coagent_list[-1].parameters(), lr=self.alpha))
            self.coagent_layers.append(temp_coagent_list)
            
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

    def coagent_forward(self,state, deterministic_runs=False):
        """ 
        sample from state based on epsilon-greedy or softmax policy for each coagent.
        """
        for j,coagent_layer in enumerate(self.coagent_layers):
            self.coagent_layers_outs[j] = np.zeros((state.shape[0],self.num_coagents)) # list of outsputs for each coagent layer
            
            for i, c in enumerate(coagent_layer):       
                if j==0:
                    q_values_coagent = c(state)  # forward pass with the i^th coagent in the given layer
                 
                else:     
                    state_coagent = torch.from_numpy(self.coagent_layers_outs[j-1]).float()
                    q_values_coagent = c(state_coagent)

                if self.exploration == "epsilon-greedy":
                    if deterministic_runs:
                        action = torch.argmax(q_values_coagent,1)  
                    else:
                        if np.random.random() > self.epsilon:
                            action = torch.argmax(q_values_coagent,1)    
                        else:
                            action = torch.from_numpy(np.random.choice(c.num_actions, q_values_coagent.shape[0]))

                        

                # choose action probabilistically, weighted by softmax(q_values)
                if self.exploration == "softmax":
                    if deterministic_runs:
                        action = torch.argmax(q_values_coagent,1) 
                    else:
                        prob = self.boltzmann(q_values_coagent)
                        action = np.zeros(q_values_coagent.shape[0])
                        for g in range(q_values_coagent.shape[0]):
                            action[g] = np.random.choice(range(q_values_coagent.shape[1]), 1, self.boltzmann(q_values_coagent)[g,:].tolist())


                c.current_q = q_values_coagent.gather(1, action.view(-1,1)).squeeze()
                self.coagent_layers_outs[j][:,i] = action              
        # main agent
        self.state_main_agent = torch.from_numpy(self.coagent_layers_outs[j]).float()
        q_values = self.model(self.state_main_agent) # shape: (num of images in batch, num_classes)
        
        return q_values

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
        
        """
        Coagents 
        again, works only for 1 layer so far (j=0)
        """
        for j, coagent_layer in enumerate(self.coagent_layers):
            for i, (c, optimizer_coagent) in enumerate(zip(coagent_layer, self.optimizers)):

                optimizer_coagent.zero_grad()
                if j==0:
                    if i!=len(coagent_layer)-1: 
                        c_q_next = torch.max(coagent_layer[i+1](batch_x),1)[0]
                    c_q_current = coagent_layer[i].current_q

                else:  # only implemented for Monte-Q, not for the other 3 versions
                    c_q_values = c(torch.from_numpy(self.coagent_layers_outs[j-1]).float()) # c's forward pass of the previous layer's output, transformed to pytorch tensor
                    c_q_current = torch.max(c_q_values, 1)[0]  

                if i==len(coagent_layer)-1:
                    coagent_loss = c.criterion(c_q_current, delta) # MSELoss
                else:
                    coagent_loss = c.criterion(c_q_current, (delta+c_q_next)/2)

                coagent_loss.backward()
                optimizer_coagent.step()
        
        if self.running_loss is None:
            self.running_loss = loss.item()
        
        self.running_loss = self.running_loss * 0.99 + .01 * loss.item()
        if self.decrement:
            self.epsilon = self.decrement_epsilon()
        return loss.item()

class Coagent(nn.Module):
    """
    Coagents
    """
    def __init__(self, params, coagent_layer_num):
        super(Coagent, self).__init__()
 
        self.in_features = params['in_features']
        self.num_classes = params['num_classes']
        self.num_coagents = params['units_layer']
        self.model_specs = params['model_specification']
        self.main_model = nn.Sequential()
        self.num_layers = self.model_specs["num_layers"]

        self.criterion = nn.MSELoss()
        self.num_actions = params['num_outputs'] # binary activation/action - temporary change
        self.coagent_out = None

        self.num_nodes_layer = None
        self.num_nn_layers = 0
        # self.num_nn_layers = params['num_nn_layers']  unused for now, as we are always testing a linear layer
        # if self.num_nn_layers > 0 :
        #     self.num_nodes_layer = 8
        """
        current implementation assumes subsequent coagent layers' input is the previous coagent layer's output,
        but in the future we could consider
        concatenating the output of the coagents with the initial input
        """
        if coagent_layer_num ==0:
            make_nn_layers(self.main_model, self.in_features, self.num_actions, self.num_nn_layers, self.num_nodes_layer)
        else:
            make_nn_layers(self.main_model, [self.num_coagents], self.num_actions, self.num_nn_layers, self.num_nodes_layer)

        self.current_q = None 

    def forward(self, x):
        q_values = self.main_model(x)
        
        return q_values

class Policy(nn.Module):
    """
    The final classifier
    """
    def __init__(self, params ):
        super(Policy, self).__init__()

        self.num_classes = params['num_classes']
        self.model_specs = params['model_specification']
        self.num_coagents = params['units_layer']
        self.model = nn.Sequential()
        self.num_layers = self.model_specs["num_layers"]

        self.num_nodes_layer = None
        self.num_nn_layers = 0
        # self.num_nn_layers = params['num_nn_layers']  unused for now, as we are always testing a linear layer
        # if self.num_nn_layers > 0 :
        #     self.num_nodes_layer = 8

        make_nn_layers(self.model, [self.num_coagents], self.num_classes, self.num_nn_layers, self.num_nodes_layer)

    def forward(self, x):
        q_values = self.model(x)   
        # return a list with the probability of each action over the action space
        
        return q_values