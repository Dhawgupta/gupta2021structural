'''
This is a the running average loss critic - standard
'''
import numpy as np
import copy
from src.agents.agent_template import ClassificationAgent
from src.agents.coagent_discrete import CoagentDiscrete
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer

MIN_DENOM = 1e-5


class CoagentDiscreteGlobalBaseline(CoagentDiscrete):
    '''
    This Agent does not use a critic as of now
    '''
    def __init__(self, params):
        super(CoagentDiscreteGlobalBaseline, self).__init__(params)
        self.baseline = None  # will be moving averaged based critic
        self.baseline_decay = params.get('baseline_decay', 0.99)
        self.update_baseline = params.get('update_baseline', True)

    def train(self, batch_x, batch_y):
        y_hat, self.coagent_states = self.network(batch_x, greedy=False)

        # criterion = nn.CrossEntropyLoss()
        criterion = self.get_objective()()
        # self.optimizer.zero_grad()
        losses = []
        loss = criterion(y_hat, batch_y)  # this is the negative reward

        with torch.no_grad():
            # delta_loss = nn.CrossEntropyLoss(reduce=False)
            delta_loss = self.get_objective()(reduce=False)
            actual_reward = -delta_loss(y_hat, batch_y)
            if self.baseline is not None:
                delta = actual_reward - self.baseline
            else:
                delta = actual_reward

            # delta = - delta_loss(class_values, batch_y).unsqueeze(1) # use the negative value of loss as reward

        coagent_loss = torch.tensor([0.0])

        coagent_state_int = [k.long() for k in self.coagent_states]

        for i in range(self.network.num_coagent_layers()):
            if i != 0:
                batch_x = self.coagent_states[i - 1]

            pi_all = self.network.get_forward_softmax(model=self.network.layers[i], x=batch_x)

            mask2 = coagent_state_int[i].long().unsqueeze(2)
            pi_a = torch.gather(pi_all, 2, mask2).squeeze(2)

            # pi_a = torch.masked_select(pi_all , mask = mask).view(pi_all.shape[0], -1)
            log_pi_a = torch.log(pi_a + MIN_DENOM)
            coagent_loss += (- (log_pi_a * delta).mean(dim=0)).sum()  # take a mean over the batch

        coagent_loss += loss
        self.optimizer.zero_grad()
        coagent_loss.backward()

        if self.gradient_clipping == 'none' or self.gradient_clipping <= 0.0:
            pass
        else:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)

        self.optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            if self.update_baseline:
                if self.baseline is None:
                    self.baseline = - loss.item()
                self.baseline = self.baseline * self.baseline_decay + (1 - self.baseline_decay) * (-loss.item())

        return losses