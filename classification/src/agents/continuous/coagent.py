import copy
from src.agents.agent_template import ClassificationAgent
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer


class Coagent(ClassificationAgent):
    ''' Plain Continous Coagent'''
    def __init__(self, params):
        super(Coagent, self).__init__(params)
        self.num_coagents = params['units_layer']
        self.nettype = params.get('model', 'generalcoagent')
        self.eval_greedy = params.get('eval_greedy', False)
        self.gradient_clipping = params.get('gradient_clipping', "none")
        model_params = copy.deepcopy(params)
        self.network = get_model("continuous",self.nettype)(model_params).to(self.device)
        # register parameters
        self.optimizer = get_optimizer(self.optim_type)(self.network.parameters(), lr=self.alpha)
        self.coagent_states = None
        self.coagent_preactivations = None
        self.coagent_states_log_probs = None  # Not being used

    def get_prediction_class_probs(self, x):
        with torch.no_grad():
            class_probs = self.forward(x)
            return class_probs

    def get_predicition_class(self, x):
        # inferencing stage i.e. target net in some sense.
        with torch.no_grad():
            return torch.max(self.forward(x, greedy=self.eval_greedy).data, 1)[1]

    def forward(self, batch_x, greedy=False):
        pi_mean, _, _ = self.network(batch_x, greedy=greedy)
        if greedy:
            return f.softmax(pi_mean, dim = 1)
        else:
            pi_std = self.network.std
            n = Normal(pi_mean, pi_std)
            class_values = n.sample()
            return f.softmax(class_values, dim = 1)



    def train(self, batch_x, batch_y):
        # non greedy eval while training
        pi_mean, self.coagent_states, self.coagent_preactivations = self.network(batch_x, greedy=False)
        pi_std = self.network.std
        losses = []
        pi = Normal(pi_mean, pi_std)
        pi_action = pi.sample()
        log_prob = pi.log_prob(pi_action)

        with torch.no_grad():
            delta_loss = nn.CrossEntropyLoss(reduce= False)
            actual_reward = - delta_loss(pi_action, batch_y).unsqueeze(1)

        losses.append( - actual_reward.mean().item())

        coagent_loss = ( - (log_prob * actual_reward)).mean(dim = 0).sum()

        for i in range(self.network.num_coagent_layers()):
            if i != 0:
                batch_x = self.coagent_states[i - 1]
            #
            reward = actual_reward
            pi_mean = self.network.get_forward_mean(model_idx = i, x = batch_x)
            pi_std = self.network.std
            n = Normal(pi_mean, pi_std)
            log_prob = n.log_prob(self.coagent_preactivations[i])
            coagent_loss_layer = (- (log_prob * reward)).mean(dim=0).sum()
            coagent_loss += coagent_loss_layer
            with torch.no_grad():
                losses.append(coagent_loss_layer.item())


        self.optimizer.zero_grad()
        coagent_loss.backward()
        if self.gradient_clipping == 'none' or self.gradient_clipping <= 0.0:
            pass
        else:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)
        self.optimizer.step()

        return losses


    def train_misc(self, loss):
        '''
        THis will train the misc variables and not eh main netrwoorks
        '''
        # no misc variables to train over here
        return None



