import numpy as np
import copy
from src.agents.discrete.coagent import Coagent
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer
import time

MIN_DENOM = 1e-5
class CoagentStaged(Coagent):
    '''
    Staged learning for coagent
    '''
    def __init__(self, params):
        super(CoagentStaged, self).__init__(params)
        self.layer_behaviour_greedy = [False for i in range(self.network.num_coagent_layers())]

    def train(self, batch_x, batch_y):
        # print(self.layer_behaviour_greedy)
        class_values, self.coagent_states = self.network(batch_x, greedy = False, greedy_layers = self.layer_behaviour_greedy)
        criterion = nn.CrossEntropyLoss()
        losses = []
        loss = criterion(class_values, batch_y) # this is the negative reward

        with torch.no_grad():
            delta_loss = nn.CrossEntropyLoss(reduce=False)
            delta = - delta_loss(class_values, batch_y).unsqueeze(1) # use the negative value of loss as reward

        coagent_loss = torch.tensor([0.0 ])

        coagent_state_int = [k.long() for k in self.coagent_states]

        for i in range(self.network.num_coagent_layers()):
            # if layer greedy no train
            if not self.layer_behaviour_greedy[i]:
                if  i != 0:
                    batch_x = self.coagent_states[i-1]
                pi_all = self.network.get_forward_softmax(model = self.network.layers[i], x = batch_x)
                mask2 = coagent_state_int[i].long().unsqueeze(2)
                pi_a = torch.gather(pi_all, 2, mask2).squeeze(2)
                log_pi_a = torch.log(pi_a + MIN_DENOM)
                coagent_loss += (- (log_pi_a * delta).mean(dim = 0) ).sum()# take a mean over the batch

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


