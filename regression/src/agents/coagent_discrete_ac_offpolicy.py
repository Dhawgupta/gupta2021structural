'''
Critic : Per Layer Critic
Credit Assignment : Boostrapping
'''
# The Actor Critic variant of the code.
import numpy as np
import copy
from src.agents.coagent_discrete_ac import CoagentDiscreteAC
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer
import time

MIN_DENOM = 1e-5


class CoagentDiscreteACOffPolicy(CoagentDiscreteAC):
    '''
    This Agent does not use a critic as of now
    '''
    def __init__(self, params):
        super(CoagentDiscreteACOffPolicy, self).__init__(params)

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
                    action_x_1 = self.network.get_forward_softmax(self.network.layers[i + 1], state_x_1).max(dim=2)[1].float()
                    # action_x_1 = self.coagent_states[i+1]
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

