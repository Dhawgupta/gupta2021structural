from src.models.general_neuralnet import GeneralNeuralNet

# continous agents
from src.models.continuous.general_coagent import GeneralCoagent as ContGeneralCoagent
from src.models.continuous.general_coagent_partition import GeneralCoagentPartition as  ContGeneralCoagentPartition

# discrete agents
from src.models.discrete.general_coagent import GeneralCoagent as DisGeneralCoagent
from src.models.discrete.general_coagent_multiaction import GeneralCoagentMultiAction as DisGeneralCoagentMultiAction
from src.models.discrete.general_coagent_subset import GeneralCoagentSubset as DisGeneralCoagentSubset
from src.models.discrete.general_coagent_staged import GeneralCoagentStaged as DisGeneralCoagentStaged
def get_model(node, model_name):

    if model_name == 'generalneuralnet':
        return GeneralNeuralNet


    # continous nodes
    if node == 'continuous':
        if model_name == 'generalcoagent':
            return ContGeneralCoagent
        if model_name == 'generalcoagentparitition':
            return ContGeneralCoagentPartition


    # discrete nodew
    if node == 'discrete':
        if model_name == 'generalcoagent':
            return DisGeneralCoagent
        if model_name == 'generalcoagentmultiaction':
            return DisGeneralCoagentMultiAction
        if model_name == 'generalcoagentsubset':
            return DisGeneralCoagentSubset
        if model_name == 'generalcoagentstaged':
            return DisGeneralCoagentStaged


    else:
        raise NotImplementedError
