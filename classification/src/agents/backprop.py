'''
This uses a general neural network class when compared to vanilla_classifier
'''
from src.agents.agent_template import ClassificationAgent
import torch
import torch.nn as nn
import torch.nn.functional as f
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer


class BackProp(ClassificationAgent):
    def __init__(self, params):
        super(BackProp, self).__init__(params)
        self.gradient_clipping = params.get('gradient_clipping', "none")
        self.network = get_model("continuous",'generalneuralnet')(params).to(self.device)
        self.optimizer = get_optimizer(self.optim_type)(self.network.parameters(), lr=self.alpha)
        self.running_loss = None

    def get_prediction_class_probs(self, x):
        with torch.no_grad():
            return self.forward(x) 

    def forward(self, x):
        class_probs = f.softmax(self.network(x), dim = 1)
        return class_probs

    def get_predicition_class(self, x):
        with torch.no_grad():
            return torch.max(self.forward(x).data, 1)[1]

    def train(self, batch_x, batch_y):
        class_values = self.network(batch_x)
        criterion = nn.CrossEntropyLoss()
        self.optimizer.zero_grad()
        loss = criterion(class_values, batch_y)
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

