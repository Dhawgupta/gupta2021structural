import copy
from src.agents.coagent import Coagent
import torch
import torch.nn as nn
from torch.distributions.normal import Normal



class CoagentGlobalBaseline(Coagent):
    ''' Plain Continous Coagent'''
    def __init__(self, params):
        super(CoagentGlobalBaseline, self).__init__(params)
        self.baseline = None  # will be moving averaged based critic
        self.baseline_decay = params.get('baseline_decay', 0.99)
        self.update_baseline = params.get('update_baseline', True)

    def train(self, batch_x, batch_y):
        losses = []
        yhatmean, self.coagent_states, self.coagent_preactivations = self.network(batch_x, greedy=False)
        pi = Normal(yhatmean, self.network.std)
        yhat = pi.sample()
        log_prob = pi.log_prob(yhat)

        with torch.no_grad():
            delta_loss = nn.MSELoss(reduce=False)
            actual_reward = - delta_loss(yhat, batch_y)
            if self.baseline is not None:
                delta = actual_reward - self.baseline
            else:
                delta = actual_reward

        coagent_loss = (- (log_prob * delta)).mean(dim=0).sum()
        with torch.no_grad():
            losses.append(- actual_reward.mean().item())
            losses.append(coagent_loss.item())

        for i in range(self.network.num_coagent_layers()):
            if i != 0:
                batch_x = self.coagent_states[i - 1]
            # update critic if using

            pi_mean = self.network.get_forward_mean(model_idx=i, x=batch_x)
            pi_std = self.network.std
            n = Normal(pi_mean, pi_std)
            log_prob = n.log_prob(self.coagent_preactivations[i])
            coagent_loss_layer = (- (log_prob * delta)).mean(dim=0).sum()
            coagent_loss += coagent_loss_layer
            losses.append(coagent_loss_layer.item())

        # no need for the layer check anymore things work just fine
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

        return losses


    def train_misc(self, loss):
        # update the global baseline
        with torch.no_grad():
            if self.baseline is None:
                self.baseline = - loss
            self.baseline = self.baseline * self.baseline_decay + (1 - self.baseline_decay) * (-loss)

        return None

