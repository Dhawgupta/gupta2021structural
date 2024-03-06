import copy
from src.agents.coagent_backprop import CoagentBackprop
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer


class CoagentGlobalBaselineBackprop(CoagentBackprop):
    ''' Plain Continous Coagent'''
    def __init__(self, params):
        super(CoagentGlobalBaselineBackprop, self).__init__(params)
        self.baseline = None  # will be moving averaged based critic
        self.baseline_decay = params.get('baseline_decay', 0.99)
        self.update_baseline = params.get('update_baseline', True)

    def train(self, batch_x, batch_y):

        pi_mean = self.network(batch_x)
        pi_std = self.std
        losses = []

        # sample stuff
        pi = Normal(pi_mean, pi_std)
        yhat = pi.sample()
        log_prob = pi.log_prob(yhat)


        with torch.no_grad():
            delta_loss = nn.MSELoss(reduce=False)
            actual_reward = - delta_loss(yhat, batch_y)  # use the negative value of loss as reward
            if self.baseline is not None:
                delta = actual_reward - self.baseline
            else:
                delta = actual_reward

        coagent_loss = ( - (log_prob * delta)).mean(dim = 0).sum()
        self.optimizer.zero_grad()
        coagent_loss.backward()
        if self.gradient_clipping == 'none':
            pass
        else:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)
        self.optimizer.step()

        with torch.no_grad():
            criterion = nn.MSELoss()
            loss = criterion(yhat, batch_y)
            if self.update_baseline:
                if self.baseline is None:
                    self.baseline = - loss.item()
                self.baseline = self.baseline * self.baseline_decay + (1 - self.baseline_decay) * (-loss.item())

        losses.append(loss.item())
        losses.append(coagent_loss.item())

        return losses


    def train_misc(self, loss):
        '''
        THis will train the misc variables and not eh main netrwoorks
        '''
        # no misc variables to train over here
        with torch.no_grad():
            if self.baseline is None:
                self.baseline = - loss
            self.baseline = self.baseline * self.baseline_decay + (1 - self.baseline_decay) * (-loss)

        return None



