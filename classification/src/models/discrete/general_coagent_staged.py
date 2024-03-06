'''
This model is just designed to allow some layers to act greedy if required
'''
import torch
from src.models.discrete.general_coagent import GeneralCoagent


class GeneralCoagentStaged(GeneralCoagent):
    def __init__(self, params):
        super(GeneralCoagentStaged, self).__init__(params)

    def sample_all_coagent_states(self, x, greedy = False, greedy_layers = []):
        # sample a full bactch of coiagent states
        states = []
        state = x
        self.all_softmax = []
        for i in range(self.num_layers):
            state_softmax_probs = self.get_forward_softmax(self.layers[i], state)
            self.all_softmax.append(state_softmax_probs)
            if greedy :
                state = state_softmax_probs.max(dim = 2)[1].float()
            else:
                if greedy_layers[i]: # added a layer greedy
                    state = state_softmax_probs.max(dim=2)[1].float()
                else:
                    state = self.sample_state(state_softmax_probs)
            states.append(state)
        return states

    def forward(self, x, greedy = False, greedy_layers = []):
        with torch.no_grad():
            states = self.sample_all_coagent_states(x, greedy = greedy, greedy_layers = greedy_layers)
        return self.layers[-1](states[-1]) , states # removed the softmax for now


