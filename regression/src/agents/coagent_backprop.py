import copy
from src.agents.agent_template import RegressionAgent
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer


class CoagentBackprop(RegressionAgent):
    ''' Plain Continous Coagent'''
    def __init__(self, params):
        super(CoagentBackprop, self).__init__(params)
        self.units_layer = params['units_layer']
        self.nettype = params.get('model', 'generalneuralnet')
        self.std = params.get('model_std', 0.1)
        self.eval_greedy = params.get('eval_greedy', False)
        self.gradient_clipping = params.get('gradient_clipping', "none")
        model_params = copy.deepcopy(params)
        self.network = get_model(self.nettype)(model_params).to(self.device)
        # register the parameters :
        self.optimizer = get_optimizer(self.optim_type)(self.network.parameters(), lr=self.alpha)


    def evaluate(self, X,y):
        with torch.no_grad():
            yhat = self.forward(X, greedy = self.eval_greedy)
            criter = nn.MSELoss()
            loss = criter(yhat, y)
            return loss.item()


    def forward(self, batch_x, greedy=False):
        pi_mean = self.network(batch_x)
        if greedy:
            # return f.softmax(pi_mean, dim = 1)
            return pi_mean
        else:
            # sample stuff
            pi_std = self.std
            n = Normal(pi_mean, pi_std)
            y_hat = n.sample()
            return y_hat


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

        coagent_loss = ( - (log_prob * actual_reward)).mean(dim = 0).sum()
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
        losses.append(loss.item())
        losses.append(coagent_loss.item())

        return losses


    def train_misc(self, loss):
        '''
        THis will train the misc variables and not eh main netrwoorks
        '''
        # no misc variables to train over here
        return None



