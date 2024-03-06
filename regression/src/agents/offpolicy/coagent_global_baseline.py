import copy
from src.agents.offpolicy.coagent import CoagentOffPAC
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer


class CoagentGlobalBaselineOffPAC(CoagentOffPAC):
    ''' Continous Coagent with Off policyu DPG kind of implementation'''
    def __init__(self, params):
        super(CoagentGlobalBaselineOffPAC, self).__init__(params)
        self.layer_baselines = []
        self.baseline_decay = params.get('baseline_decay', 0.99)
        self.update_baseline = params.get('update_baseline', True)

        for i in range(self.network.num_coagent_layers() + 1):
            self.layer_baselines.append(None)

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
            reward = actual_reward
            if self.layer_baselines[-1] is not None:
                delta = reward - self.layer_baselines[-1]
            else:
                delta = reward
            # update baseline
            if self.update_baseline:
                total_loss = reward.mean().item()
                if self.layer_baselines[-1] is None:
                    self.layer_baselines[-1] = total_loss
                self.layer_baselines[-1] = self.layer_baselines[-1] * self.baseline_decay + (1 - self.baseline_decay) * (total_loss)

        coagent_loss = ( - (log_prob * delta)).mean(dim = 0).sum()
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
                pi = Normal(target_policy_actions_mean, self.std_tar)  # use the target policys
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
                reward = critic_target
                if self.layer_baselines[i] is not None:
                    delta = reward - self.layer_baselines[i]
                else:
                    delta = reward
                if self.update_baseline:
                    total_loss = reward.mean().item()
                    if self.layer_baselines[i] is None:
                        self.layer_baselines[i] = total_loss
                    self.layer_baselines[i] = self.layer_baselines[i] * self.baseline_decay + (1 - self.baseline_decay) * (total_loss)

            pi_mean = self.network.get_forward_mean(model_idx = i, x = data_x)
            pi_std = self.std_tar
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

        self.update_std()
        return losses


    def train_misc(self, loss):
        '''
        THis will train the misc variables and not eh main netrwoorks
        '''
        # no misc variables to train over here
        return None