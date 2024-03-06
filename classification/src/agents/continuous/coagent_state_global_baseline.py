from src.agents.continuous.coagent import Coagent
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.optimizers.registry import get_optimizer



class CoagentStateGlobalBaseline(Coagent):
    ''' Plain Continous Coagent'''
    def __init__(self, params):
        super(CoagentStateGlobalBaseline, self).__init__(params)
        # define a global balue function and optimizer for that
        self.global_baseline = nn.Linear(params['in_features'][0], 1)
        # init stuff to zero
        self.global_baseline.weight.data.fill_(0.0)
        self.global_baseline.bias.data.fill_(0.0)
        self.global_baseline_optimizer = get_optimizer(self.optim_type)(self.global_baseline.parameters(),lr=self.alpha)


    def train(self, batch_x, batch_y):
        losses = []
        yhatmean, self.coagent_states, self.coagent_preactivations = self.network(batch_x, greedy=False)
        pi = Normal(yhatmean, self.network.std)
        yhat = pi.sample()
        log_prob = pi.log_prob(yhat)
        state_values = self.global_baseline(batch_x)
        with torch.no_grad():
            delta_loss = nn.CrossEntropyLoss(reduce=False)
            actual_reward = - delta_loss(yhat, batch_y).unsqueeze(1)
            delta = actual_reward - state_values

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

        # update the value function
        vb_criterion = nn.MSELoss()
        vb_loss = vb_criterion(state_values, actual_reward)
        self.global_baseline_optimizer.zero_grad()
        vb_loss.backward()
        self.global_baseline_optimizer.step()
        losses.append(vb_loss.item())

        return losses

    def train_misc(self, loss):
        # update the global baseline
        with torch.no_grad():
            if self.baseline is None:
                self.baseline = - loss
            self.baseline = self.baseline * self.baseline_decay + (1 - self.baseline_decay) * (-loss)

        return None
