import numpy as np
import copy
from src.agents.agent_template import ClassificationAgent
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer
import time
actionPolicy = namedtuple('actionPolicy', ['action', 'action_prob'])
MIN_DENOM = 1e-5
class CoagentMultiAction(ClassificationAgent):
    '''
    This Agent does not use a critic as of now
    '''
    def __init__(self, params):
        super(CoagentMultiAction, self).__init__(params)
        self.nettype = params.get('model', 'generalcoagentmultiaction')
        self.num_coagents = params['units_layer']
        self.eval_greedy = params.get('eval_greedy', False)
        self.gradient_clipping = params.get('gradient_clipping', "none")
        model_params = copy.deepcopy(params)
        self.network = get_model("discrete", self.nettype)(model_params).to(self.device)
        self.optimizer = get_optimizer(self.optim_type)(self.network.parameters(), lr = self.alpha)
        self.coagent_states = None
        self.coagent_actions = None



    def get_prediction_class_probs(self, x):
        with torch.no_grad():
            class_probs = self.forward(x)
            return class_probs


    def get_predicition_class(self, x):
        with torch.no_grad():
            return torch.max(self.forward(x, greedy = self.eval_greedy).data, 1)[1]


    def forward(self, batch_x, greedy = False):
        prediction_y , self.coagent_states, self.coagent_actions = self.network(batch_x, greedy = greedy)
        # final sample from prediciton over here
        # FIXME the sampling over here
        # we can;t sample from discrete because thats only 0 and 1 and making it continous would required a gaussina distribution
        prediction_y = f.softmax(prediction_y, dim = 1)
        return prediction_y

    def train(self, batch_x, batch_y):
        # class_probs = self.forward(batch_x)
        class_values, self.coagent_states, self.coagent_actions = self.network(batch_x, greedy = False)

        criterion = nn.CrossEntropyLoss()
        # self.optimizer.zero_grad()
        losses = []
        loss = criterion(class_values, batch_y) # this is the negative reward


        with torch.no_grad():
            delta_loss = nn.CrossEntropyLoss(reduce=False)
            delta = - delta_loss(class_values, batch_y).unsqueeze(1) # use the negative value of loss as reward

        coagent_loss = torch.tensor([0.0 ])
        

        coagent_action_int = [k.long() for k in self.coagent_actions]

        for i in range(self.network.num_coagent_layers()):
            if  i != 0:
                batch_x = self.coagent_states[i-1]
                # action_x = self.coagent_actions[i-1]
            
            pi_all = self.network.get_forward_softmax(model = self.network.layers[i], x = batch_x)

            # Vectorized Version

            mask = coagent_action_int[i].long().unsqueeze(2)
            pi_a = torch.gather(pi_all, 2, mask ).squeeze(2)

            # pi_a = torch.masked_select(pi_all , mask = mask).view(pi_all.shape[0], -1)
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


