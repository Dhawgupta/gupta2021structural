from src.models.general_neuralnet import GeneralNeuralNet
from src.models.general_coagent import GeneralCoagent
from src.models.general_coagent_partition import GeneralCoagentPartition
from src.models.general_coagent_discrete import GeneralCoagentDiscrete

def get_model(model_name):
    if model_name == 'generalneuralnet':
        return GeneralNeuralNet
    if model_name == 'generalcoagent':
        return GeneralCoagent
    if model_name == 'generalcoagentparitition':
        return GeneralCoagentPartition
    if model_name == 'generalcoagentdiscrete':
        return GeneralCoagentDiscrete
    else:
        raise NotImplementedError
