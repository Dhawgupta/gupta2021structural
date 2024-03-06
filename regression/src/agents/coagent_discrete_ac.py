'''
Critic : Per Layer Critic
Credit Assignment : Boostrapping
'''
# The Actor Critic variant of the code.
import numpy as np
import copy
from src.agents.coagent_discrete import CoagentDiscrete
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer
import time

MIN_DENOM = 1e-5

def define_network(in_features, nodes, output, num_layers = 0):
    # define a critic network
    if num_layers == 0:
        model =  nn.Linear(in_features, output)

        # model.weight.data.fill_(0.0)
        # model.bias.data.fill_(0.0)

    else:
        model = nn.Sequential()
        model.add_module("input-layer",nn.Linear(in_features, nodes))
        model.add_module("input-activation", nn.ReLU())
        for i in range(num_layers-1):
            model.add_module(f"{i}-layer",nn.Linear(nodes, nodes))
            model.add_module(f"{i}-activation",nn.ReLU())

        model.add_module(f"{num_layers}-layer",nn.Linear(nodes, output))
    return model

class CoagentDiscreteAC(CoagentDiscrete):
    '''
    This Agent does not use a critic as of now
    '''
    def __init__(self, params):
        super(CoagentDiscreteAC, self).__init__(params)
        self.critic_layers = params.get('critic_layer', 1)
        self.critic_nodes = params.get('critic_nodes', 16)
        self.layer_critic = []
        self.layer_critic_optimizers = []
        self.critic_alpha = params.get('critic_alpha_ratio', 2) * self.alpha # the ratio of the learning rates

        for i in range(self.network.num_coagent_layers()):
            if i == 0: # the first layer
                self.layer_critic.append( define_network(params['in_features'] + self.num_coagents,self.critic_nodes, 1, self.critic_layers ) )
            else:
                self.layer_critic.append( define_network(self.num_coagents *2, self.critic_nodes, 1, self.critic_layers))

            self.layer_critic_optimizers.append(get_optimizer(self.optim_type)(self.layer_critic[-1].parameters(), lr  = self.critic_alpha))

    def train(self, batch_x, batch_y):
        y_hat, self.coagent_states = self.network(batch_x, greedy = False)

        criterion = self.get_objective()()
        # self.optimizer.zero_grad()
        losses = []
        loss = criterion(y_hat, batch_y) # this is the negative reward


        with torch.no_grad():
            # delta_loss = nn.CrossEntropyLoss(reduce=False)
            delta_loss = self.get_objective()(reduce = False)
            delta = -delta_loss(y_hat, batch_y)

        coagent_loss = torch.tensor([0.0 ])
        coagent_state_int = [k.long() for k in self.coagent_states]

        for i in range(self.network.num_coagent_layers() - 1, -1, -1):
            if i != 0:
                data_x = self.coagent_states[i-1] # FIXME check if correct state is selected or now
            else:
                data_x = batch_x

            with torch.no_grad():
                action = self.coagent_states[i] # get the action
                input_x = torch.cat((data_x, action), dim = 1)

            critic_output = self.layer_critic[i](input_x)

            if i == self.network.num_coagent_layers() - 1:
                critic_target = delta
            else:
                with torch.no_grad():
                    state_x_1 = self.coagent_states[i]
                    action_x_1 = self.coagent_states[i+1]
                    input_x_1 = torch.cat((state_x_1, action_x_1), dim= 1)
                    critic_target = self.layer_critic[i+1](input_x_1)

            mse_loss = nn.MSELoss()
            critic_loss = mse_loss(critic_output, critic_target)
            self.layer_critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.layer_critic_optimizers[i].step()
            with torch.no_grad():
                reward = critic_target

            pi_all = self.network.get_forward_softmax(model = self.network.layers[i], x = data_x)
            mask2 = coagent_state_int[i].long().unsqueeze(2)
            pi_a = torch.gather(pi_all, 2, mask2).squeeze(2)
            log_pi_a = torch.log(pi_a + MIN_DENOM)
            coagent_loss += (- (log_pi_a * reward).mean(dim = 0) ).sum()# take a mean over the batch
        coagent_loss += loss
        self.optimizer.zero_grad()
        coagent_loss.backward()

        if self.gradient_clipping == 'none' or self.gradient_clipping <= 0.0:
            pass
        else:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)
        self.optimizer.step()
        losses.append(loss.item())
        return losses

