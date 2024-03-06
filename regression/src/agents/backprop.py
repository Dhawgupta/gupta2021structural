'''
This uses a general neural network class when compared to vanilla_classifier
'''
from src.agents.agent_template import RegressionAgent
import torch
import torch.nn as nn
import torch.nn.functional as f
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer


class BackProp(RegressionAgent):
    def __init__(self, params):
        super(BackProp, self).__init__(params)
        self.gradient_clipping = params.get('gradient_clipping', "none")
        self.network = get_model('generalneuralnet')(params).to(self.device)
        if self.optim_type == 'rmsprop':
            self.optimizer = get_optimizer(self.optim_type)(self.network.parameters(), lr=self.alpha, alpha=self.RMSpropBeta)
        elif self.optim_type == 'adam':
            self.optimizer = get_optimizer(self.optim_type)(self.network.parameters(), lr=self.alpha, betas=(self.AdamBeta1, self.AdamBeta2))


    def forward(self, x):
        yhat = self.network(x)
        return yhat

    def evaluate(self, X, y):
        with torch.no_grad():
            yhat = self.network(X)
            criterion = nn.MSELoss()
            loss = criterion(yhat, y)
            return loss.item()

    def train(self, batch_x, batch_y):
        yhat = self.network(batch_x)
        criterion = nn.MSELoss()
        self.optimizer.zero_grad()
        loss = criterion(yhat, batch_y)
        loss.backward()
        losses = []
        if self.gradient_clipping == 'none':
            pass
        else:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)
        self.optimizer.step()
        losses.append(loss.item())
        return losses

    def train_misc(self, loss):
        return None

