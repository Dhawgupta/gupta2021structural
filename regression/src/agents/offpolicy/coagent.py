import copy
from src.agents.agent_template import RegressionAgent
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer


class CoagentOffPAC(RegressionAgent):
    ''' Continous Coagent with Off policyu DPG kind of implementation'''
    def __init__(self, params):
        super(CoagentOffPAC, self).__init__(params)
        self.num_coagents = params['units_layer']
        self.nettype = params.get('model', 'generalcoagent')
        self.eval_greedy = params.get('eval_greedy', False)
        self.std_tar = torch.tensor(params.get('std_tar', 0.01), requires_grad=False)
        self.std_beh = torch.tensor(params.get('std_beh', 4.0), requires_grad=False)
        self.gradient_clipping = params.get('gradient_clipping', "none")
        model_params = copy.deepcopy(params)
        self.network = get_model(self.nettype)(model_params).to(self.device)
        self.network.std = self.std_beh # maybe set this # FIXME
        # register the parameters :
        self.optimizer = get_optimizer(self.optim_type)(self.network.parameters(), lr=self.alpha)
        self.coagent_states = None
        self.coagent_preactivations = None
        self.coagent_states_log_probs = None  # Not being used
        self.std_decay = params.get('std_decay', False)
        if self.std_decay:
            self.std_decay_rate = params.get('std_decay_rate' , 1.0)

        # make the layer critics:
        self.layer_critics = []
        self.layer_critics_optimizer = []
        for i in range(self.network.num_coagent_layers() + 1): # make n + 1 coagent critic layers
            if i == 0:
                self.layer_critics.append(nn.Linear(params['in_features'] + self.num_coagents, 1))
            elif i == self.network.num_coagent_layers():
                self.layer_critics.append(nn.Linear(self.num_coagents  + 1, 1))
            else:
                self.layer_critics.append(nn.Linear(self.num_coagents * 2, 1))


            # init stuff to zero
            self.layer_critics[-1].weight.data.fill_(0.0)
            self.layer_critics[-1].bias.data.fill_(0.0)
            self.layer_critics_optimizer.append(
                get_optimizer(self.optim_type)(self.layer_critics[-1].parameters(), lr=self.alpha * 2))



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
            # FIXME , select sampling from target policy
            dist = Normal(yhatmean, self.std_tar )
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
        yhatmean , self.coagent_states, self.coagent_preactivations = self.network(batch_x, greedy = False)
        # pi = Normal(yhatmean, self.network.std)
        pi = Normal(yhatmean, self.std_beh) # use the bheavioru policy
        yhat = pi.sample()
        pi_tar = Normal(yhatmean, self.std_tar)
        log_prob = pi_tar.log_prob(yhat)
        with torch.no_grad():
            # create the appropriate input
            input_x = torch.cat((self.coagent_states[-1], yhat), dim =1)
        critic_output = self.layer_critics[-1](input_x)

        # yhat is the action
        with torch.no_grad():
            delta_loss = self.get_objective()(reduce = False)
            # delta_loss = nn.MSELoss(reduce = False)
            actual_reward = - delta_loss(yhat, batch_y)
        # calculate loss
        mse_loss = nn.MSELoss()
        critic_Loss = mse_loss(critic_output, actual_reward)
        self.layer_critics_optimizer[-1].zero_grad()
        critic_Loss.backward()
        self.layer_critics_optimizer[-1].step()
        with torch.no_grad():
            # reward = self.layer_critics[-1](input_x)
            # use the reward for the final layer update
            reward = actual_reward
        coagent_loss = ( - (log_prob * reward)).mean(dim = 0).sum()
        with torch.no_grad():
            losses.append(- actual_reward.mean().item())
            losses.append(coagent_loss.item())

        for i in range(self.network.num_coagent_layers() - 1, -1, -1): # start from the second last layer

            if i != 0:
                data_x = self.coagent_states[i - 1]
            else:
                data_x = batch_x
            # update critic if using
            ## CRITIC Begins

            with torch.no_grad():
                action = self.coagent_preactivations[i]
                input_x = torch.cat((data_x, action), dim=1)

            critic_output = self.layer_critics[i](input_x)

            # build the target for this layer
            with torch.no_grad():
                state_x_1 = self.coagent_states[i]
                target_policy_actions_mean = self.network.get_forward_mean(model_idx= i+1, x = state_x_1) #layers[i+1](state_x_1)
                pi = Normal(target_policy_actions_mean, self.std_tar)  # use the target polic
                target_action = pi.sample()
                input_x_1 = torch.cat((state_x_1, target_action), dim = 1)
                critic_target = self.layer_critics[i+1](input_x_1)

            # calculate loss
            mse_loss = nn.MSELoss()
            critic_Loss = mse_loss(critic_output, critic_target)
            self.layer_critics_optimizer[i].zero_grad()
            critic_Loss.backward()
            self.layer_critics_optimizer[i].step()

            ## CRITIC update over

            with torch.no_grad():
                # reward = self.layer_critics[i](input_x)
                # use teh critic targget as the targetr
                reward = critic_target

            pi_mean = self.network.get_forward_mean(model_idx = i, x = data_x)
            pi_std = self.std_tar
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