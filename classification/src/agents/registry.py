# continous agents
from src.agents.backprop import BackProp
from src.agents.continuous.coagent import Coagent as ContCoagent
from src.agents.continuous.coagent_global_baseline import CoagentGlobalBaseline as ContCoagentGlobalBaseline
from src.agents.continuous.coagent_state_layer_baseline import CoagentStateLayerlBaseline as ContCoagentStateLayerBaseline
from src.agents.continuous.coagent_state_global_baseline import CoagentStateGlobalBaseline as ContCoagentStateGlobalBaseline

# discrete agents
from src.agents.discrete.coagent import Coagent as DisCoagents
from src.agents.discrete.coagent_global_baseline import CoagentGlobalBaseline as DisCoagentGlobalBaseline
from src.agents.discrete.coagent_multiaction import CoagentMultiAction as DisCoagentMultiAction
from src.agents.discrete.coagent_ac import CoagentAC as DisCoagentAC
from src.agents.discrete.coagent_subset import CoagentSubset as DisCoagentSubset
from src.agents.discrete.coagent_ac_offpolicy import CoagentACOffPolicy as DisCoagentACOffPolicy
from src.agents.discrete.coagent_staged import CoagentStaged as DisCoagentStaged
from src.agents.actionvalue.Monte_Classifier import Monte_Classifier as Monte_Classifier
from src.agents.actionvalue.Q_with_global_loss import Q_with_global_loss as Q_with_global_loss
from src.agents.actionvalue.Sarsa_with_global_loss import Sarsa_with_global_loss as Sarsa_with_global_loss

def get_agent(node_type, agent_name):
    # remember to provide dummy node_type for agents which are not depende on that
    if agent_name == 'backprop':
        return BackProp

    if node_type == 'continuous':
        # PG methods
        if agent_name == 'coagent':
            return ContCoagent
        if agent_name == 'coagent_global_baseline':
            return ContCoagentGlobalBaseline
        if agent_name == 'coagent_state_layer_baseline':
            return ContCoagentStateLayerBaseline
        if agent_name == 'coagent_state_global_baseline':
            return ContCoagentStateGlobalBaseline

    if node_type == 'discrete':
        # PG methods
        if agent_name == 'coagent':
            return DisCoagents
        if agent_name == 'coagent_global_baseline':
            return DisCoagentGlobalBaseline
        if agent_name == 'coagent_multiaction':
            return DisCoagentMultiAction
        if agent_name == 'coagent_ac':
            return DisCoagentAC
        if agent_name == 'coagent_subset':
            return DisCoagentSubset
        if agent_name == 'coagent_staged':
            return DisCoagentStaged
        if agent_name == 'coagent_ac_offpolicy':
            return DisCoagentACOffPolicy
        # action value methods
        if agent_name == 'Monte_Classifier':
            return Monte_Classifier
        if agent_name == 'Q_with_global_loss':
            return Q_with_global_loss
        if agent_name == 'Sarsa_with_global_loss':
            return Sarsa_with_global_loss

    else:
        raise NotImplementedError

