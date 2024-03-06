from src.agents.continuous.coagent  import Coagent
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.optimizers.registry import get_optimizer



class CoagentStateLayerlBaseline(Coagent):
    ''' Plain Continous Coagent'''
    def __init__(self, params):
        super(CoagentStateLayerlBaseline, self).__init__(params)
        self.layer_baselines = []  # list containing baselines for each  layers
        self.layer_baseline_optimizers = []
        for i in range(self.network.num_coagent_layers() + 1): # add another agent for the last stochastic layer
            if i == 0:
                self.layer_baselines.append(nn.Linear(params['in_features'][0], 1))
            else:
                self.layer_baselines.append(nn.Linear(self.num_coagents, 1))
            # init stuff to zero
            self.layer_baselines[-1].weight.data.fill_(0.0)
            self.layer_baselines[-1].bias.data.fill_(0.0)
            self.layer_baseline_optimizers.append(
                get_optimizer(self.optim_type)(self.layer_baselines[-1].parameters(), lr=self.alpha))


    def train(self, batch_x, batch_y):
        losses = []
        yhatmean, self.coagent_states, self.coagent_preactivations = self.network(batch_x, greedy=False)
        pi = Normal(yhatmean, self.network.std)
        yhat = pi.sample()
        log_prob = pi.log_prob(yhat)
        state_values = self.layer_baselines[-1](self.coagent_states[-1]) # last layer
        with torch.no_grad():
            delta_loss = nn.CrossEntropyLoss(reduce=False)
            actual_reward = - delta_loss(yhat, batch_y).unsqueeze(1)
            delta = actual_reward - state_values

        coagent_loss = (- (log_prob * delta)).mean(dim=0).sum()

        with torch.no_grad():
            losses.append(- actual_reward.mean().item())
            losses.append(coagent_loss.item())

        # update value function
        vb_criterion = nn.MSELoss()
        vb_loss = vb_criterion(state_values, actual_reward)
        self.layer_baseline_optimizers[-1].zero_grad()
        vb_loss.backward()
        self.layer_baseline_optimizers[-1].step()
        losses.append(vb_loss.item())

        for i in range(self.network.num_coagent_layers()):
            if i != 0:
                batch_x = self.coagent_states[i - 1]
            # update critic if using
            state_values = self.layer_baselines[i](batch_x)
            pi_mean = self.network.get_forward_mean(model_idx=i, x=batch_x)
            pi_std = self.network.std
            n = Normal(pi_mean, pi_std)
            log_prob = n.log_prob(self.coagent_preactivations[i])
            with torch.no_grad():
                delta = actual_reward - state_values
            coagent_loss_layer = (- (log_prob * delta)).mean(dim=0).sum()
            coagent_loss += coagent_loss_layer

            # update the value function using MonteCarlo
            vb_criterion = nn.MSELoss()
            vb_loss = vb_criterion(state_values, actual_reward)
            self.layer_baseline_optimizers[i].zero_grad()
            vb_loss.backward()
            self.layer_baseline_optimizers[i].step()
            # store stats
            losses.append(coagent_loss_layer.item())
            losses.append(vb_loss.item())

        # no need for the layer check anymore things work just fine
        self.optimizer.zero_grad()
        coagent_loss.backward()
        if self.gradient_clipping == 'none':
            pass
        else:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)
        self.optimizer.step()


        return losses

    def train_misc(self, loss):

        return None
