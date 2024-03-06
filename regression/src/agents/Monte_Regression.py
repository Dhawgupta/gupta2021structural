import numpy as np
import copy
from src.agents.agent_template import RegressionAgent
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer
from src.agents.coagent import Coagent
from torch.distributions.normal import Normal



def make_nn_layers(model, in_features, out, num_layers, num_nodes_layer = None):
    if num_layers == 0:  # linear classifier
        model.add_module("0", nn.Linear(in_features, out_features=out))
    else:
        model.add_module("0", nn.Linear(in_features, num_nodes_layer))
        model.add_module("0-a", nn.ReLU())
        for i in range(num_layers - 1):
            model.add_module(f"{i + 1}", nn.Linear(num_nodes_layer, num_nodes_layer))
            model.add_module((f"{i + 1}-a"), nn.ReLU())
        model.add_module(f"{num_layers}", nn.Linear(num_nodes_layer, out))
    return model

class Monte_Regression(Coagent):
    '''
    '''
    def __init__(self, params):
        super(Monte_Regression, self).__init__(params)
        # number of coagents
        self.units_layer = params['units_layer']
        #self.nettype = params['model']
 
        self.in_features = params['in_features']
        self.model_specs = params['model_specification']
        self.num_layers = self.model_specs['num_layers']
        self.num_nn_layers = params['num_nn_layers'] # #hidden layers

        self.running_loss = None
        self.coagent_layers = [] # will be a list of list of coagents
        self.coagent_layers_outs=[None]*self.num_layers # initialise output of each coagent layer 

        self.epsilon = int(params['epsilon'])
        self.decrement = True if self.epsilon == 1 else False
        self.eps_dec = 1e-4
        self.eps_min = 0.01    
        #self.tau = params['tau']
        self.exploration = params['exploration']
        self.eps_min_reached = False
        self.std = 0.1
        self.eval_greedy = params.get('eval_greedy', False)
        self.gradient_clipping = params.get('gradient_clipping', "none")

        #self.coagents = []
        self.optimizers = [] 
        self.criterion = nn.MSELoss()

        for j in range(self.num_layers):
            temp_coagent_list = []
            for i in range(self.units_layer):
                temp_coagent_list.append(Coagent(params,j))
                self.optimizers.append(get_optimizer(self.optim_type)(temp_coagent_list[-1].parameters(), lr=self.alpha))
            self.coagent_layers.append(temp_coagent_list)
            
        self.model = Policy(params)
        self.optimizer_model = get_optimizer(self.optim_type)(self.model.parameters(), lr=self.alpha)

    def evaluate(self, X,y):
        with torch.no_grad():
            yhat = self.coagent_forward(X, greedy = self.eval_greedy)
            criter = nn.MSELoss()
            loss = criter(yhat, y)
            return loss.item()

    def decrement_epsilon(self):       
        return self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def boltzmann(self,q_values):
        numerator = torch.exp(q_values/self.tau)
        denominator = torch.sum(torch.exp(q_values/self.tau), 0)
        return (numerator/denominator)

    def coagent_forward(self,state, greedy=False):
        """ 
        sample from state based on epsilon-greedy or softmax policy for each coagent.
        """
        for j,coagent_layer in enumerate(self.coagent_layers):
            self.coagent_layers_outs[j] = torch.zeros((state.shape[0],self.units_layer)) # list of outsputs for each coagent layer
            
            for i, c in enumerate(coagent_layer):       
                if j==0:
                    q_mean = c(state).squeeze()  # forward pass with the i^th coagent in the given layer
                else:     
                    state_coagent = self.coagent_layers_outs[j-1].float()
                    q_mean = c(state_coagent)

                if self.exploration == "epsilon-greedy":
                    if greedy:
                        action = torch.argmax(q_mean,1)  
                    else:
                        if np.random.random() > self.epsilon:
                            action = torch.argmax(q_mean,1) 
                        else:
                            # pi_std = self.std
                            # n = Normal(q_mean, pi_std)
                            # action = n.sample()
                            action = torch.from_numpy(np.random.choice(c.num_actions, q_mean.shape[0]))


                # choose action probabilistically, weighted by softmax(q_values) TODO: implement for regression
                if self.exploration == "softmax":
                    if greedy:
                        action = torch.argmax(q_mean,1) 
                    else:
                        prob = self.boltzmann(q_mean)
                        action = np.zeros(q_mean.shape[0])
                        for g in range(q_mean.shape[0]):
                            action[g] = np.random.choice(range(q_mean.shape[1]), 1, self.boltzmann(q_mean)[g,:].tolist())

                c.current_q = q_mean.gather(1, action.view(-1,1)).squeeze()
                self.coagent_layers_outs[j][:,i] = action              

        # main agent
        self.state_main_agent = self.coagent_layers_outs[j].float()
        q_mean_final = self.model(self.state_main_agent) # shape: (num of images in batch, num_classes)
        
        return q_mean_final

    def get_prediction_class_probs(self, x):
        with torch.no_grad():
            class_probs = self.coagent_forward(x)
            return class_probs

    def get_predicition_class(self, x):
        with torch.no_grad():
            return torch.max(self.coagent_forward(x).data, 1)[1]

    def train(self, batch_x, batch_y):
        """
        training loop for the coagents and the main agent
        """
        losses = []
        action_outputs = self.coagent_forward(batch_x)
        loss = self.criterion(action_outputs, batch_y) # this is the negative reward cross-entropy

        # update the main optimizer
        self.optimizer_model.zero_grad()
        loss.backward()
        self.optimizer_model.step()

        with torch.no_grad():
            delta_loss = nn.MSELoss(reduce=False)
            delta = - delta_loss(action_outputs, batch_y) # use the negative value of loss as reward



        # q_mean_final = self.coagent_forward(batch_x)
        # q_std = self.std

        # q = Normal(q_mean_final, q_std)
        # yhat = q.sample()
        # log_prob = q.log_prob(yhat)
        # # update the main optimizer


        # with torch.no_grad():
        #     delta_loss = nn.MSELoss(reduce=False)
        #     delta = - delta_loss(q_mean_final, batch_y.squeeze()) # use the negative value of loss as reward
        
        # loss = ( - (log_prob * delta)).mean(dim = 0).sum()
        
        # self.optimizer_model.zero_grad()
        # loss.backward()

        # if self.gradient_clipping == 'none':
        #     pass
        # else:
        #     nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)

        # self.optimizer_model.step()        
        """
        Coagents 
        again, works only for 1 layer so far (j=0)
        """
        for j, coagent_layer in enumerate(self.coagent_layers):
            for i, (c, optimizer_coagent) in enumerate(zip(coagent_layer, self.optimizers)):

                optimizer_coagent.zero_grad()
                if j==0:

                    c_q_current = coagent_layer[i].current_q

                else:  # only implemented for Monte-Q, not for the other 3 versions TODO implement for regression
                    c_q_values = c(torch.from_numpy(self.coagent_layers_outs[j-1]).float()) # c's forward pass of the previous layer's output, transformed to pytorch tensor
                    c_q_current = torch.max(c_q_values, 1)[0]  

                coagent_loss = c.criterion(c_q_current, delta)
                coagent_loss.backward()
                optimizer_coagent.step()
        
        if self.running_loss is None:
            self.running_loss = loss.item()
        
        self.running_loss = self.running_loss * 0.99 + .01 * loss.item()
        if self.decrement:
            self.epsilon = self.decrement_epsilon()

        losses.append(loss.item())
        losses.append(coagent_loss.item())

        return losses

class Coagent(nn.Module):
    """
    Coagents
    """
    def __init__(self, params, coagent_layer_num):
        super(Coagent, self).__init__()
 
        self.in_features = params['in_features']
        self.units_layer = params['units_layer']
        self.model_specs = params['model_specification']
        self.num_nn_layers = params['num_nn_layers']        
        self.main_model = nn.Sequential()
        self.num_layers = self.model_specs["num_layers"]
        self.num_actions = params['num_outputs']
        self.num_nodes_layer = None
        self.criterion = nn.MSELoss()
        self.coagent_out = None

        if self.num_nn_layers > 0 :
            self.num_nodes_layer = 8
        """
        current implementation assumes subsequent coagent layers' input is the previous coagent layer's output,
        but in the future we could consider
        concatenating the output of the coagents with the initial input
        """
        if coagent_layer_num ==0:
            make_nn_layers(self.main_model, self.in_features, self.num_actions, self.num_nn_layers, self.num_nodes_layer)
        else:
            make_nn_layers(self.main_model, [self.units_layer], self.num_actions, self.num_nn_layers, self.num_nodes_layer)

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

        self.model_specs = params['model_specification']
        self.units_layer = params['units_layer']
        self.num_nn_layers = params['num_nn_layers']
        self.model = nn.Sequential()
        self.num_layers = self.model_specs["num_layers"]
        self.num_nodes_layer = None

        if self.num_nn_layers > 0 :
            self.num_nodes_layer = 8

        make_nn_layers(self.model, self.units_layer, 1, self.num_nn_layers, self.num_nodes_layer)

    def forward(self, x):
        q_values = self.model(x)   
        # return a list with the probability of each action over the action space
        
        return q_values
