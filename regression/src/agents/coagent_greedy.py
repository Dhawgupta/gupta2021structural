import copy
from src.agents.agent_template import RegressionAgent
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer


class CoagentGreedy(RegressionAgent):
    ''' Plain Continous Coagent'''
    def __init__(self, params):
        super(CoagentGreedy, self).__init__(params)
        self.num_coagents = params['units_layer']
        self.nettype = params.get('model', 'generalcoagent')
        self.eval_greedy = params.get('eval_greedy', False)
        self.gradient_clipping = params.get('gradient_clipping', "none")
        model_params = copy.deepcopy(params)
        self.network = get_model(self.nettype)(model_params).to(self.device)
        # register the parameters :
        self.optimizer = get_optimizer(self.optim_type)(self.network.parameters(), lr=self.alpha)
        self.coagent_states = None
        self.coagent_preactivations = None
        self.coagent_states_log_probs = None  # Not being used
        self.std_decay = params.get('std_decay', False)
        if self.std_decay:
            self.std_decay_rate = params.get('std_decay_rate' , 1.0)



    def evaluate(self, X,y):
        '''
        Need to use greedy evaluation for the case of coagents
        '''
        with torch.no_grad():
            yhat = self.forward(X, greedy = self.eval_greedy)
            criter = self.get_objective()()
            loss = criter(yhat, y)
            return loss.item()



    def forward(self, batch_x, greedy=False):
        yhatmean , _, _ = self.network(batch_x, greedy=greedy)
        # sample from yhat
        if greedy:
            yhat =  yhatmean
        else:
            dist = Normal(yhatmean, self.network.std )
            yhat = dist.sample()

        return yhat

    def update_std(self):
        '''
        Update the standard deviation for the learners
        '''
        if self.std_decay:
            # update the
            self.network.std = max( self.network.std * self.std_decay_rate, 0.001)


    def train(self, batch_x, batch_y):
        losses = []
        # @new switch to a greedy policy
        yhatmean , self.coagent_states, self.coagent_preactivations = self.network(batch_x, greedy = True) # sample as a greedy policy
        pi = Normal(yhatmean, self.network.std)
        # switch to a greed policy
        with torch.no_grad():
            yhat = yhatmean
        # yhat = pi.sample()
        log_prob = pi.log_prob(yhat)

        with torch.no_grad():
            delta_loss = self.get_objective()(reduce = False)
            # delta_loss = nn.MSELoss(reduce = False)
            actual_reward = - delta_loss(yhat, batch_y)

        coagent_loss = ( - (log_prob * actual_reward)).mean(dim = 0).sum()
        with torch.no_grad():
            losses.append(- actual_reward.mean().item())
            losses.append(coagent_loss.item())

        for i in range(self.network.num_coagent_layers()):
            if i != 0:
                batch_x = self.coagent_states[i - 1]
            # update critic if using
            reward = actual_reward
            pi_mean = self.network.get_forward_mean(model_idx = i, x = batch_x)
            pi_std = self.network.std
            n = Normal(pi_mean, pi_std)
            log_prob = n.log_prob(self.coagent_preactivations[i])
            coagent_loss_layer = (- (log_prob * reward)).mean(dim=0).sum()
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

        self.update_std()
        return losses


    def train_misc(self, loss):
        '''
        THis will train the misc variables and not eh main netrwoorks
        '''
        # no misc variables to train over here
        return None