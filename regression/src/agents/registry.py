from src.agents.backprop import BackProp
from src.agents.coagent import Coagent
from src.agents.coagent_global_baseline import CoagentGlobalBaseline
from src.agents.coagent_backprop import CoagentBackprop
from src.agents.coagent_global_baseline_backprop import CoagentGlobalBaselineBackprop
from src.agents.Monte_Regression import Monte_Regression
from src.agents.Sarsa_with_global_loss import Sarsa_with_global_loss
from src.agents.Sarsa import Sarsa
from src.agents.Q_with_global_loss import Q_with_global_loss
from src.agents.Q import Q
from src.agents.Linear import Linear


# off policy agents
from src.agents.offpolicy.coagent import CoagentOffPAC
from src.agents.offpolicy.coagent_global_baseline import CoagentGlobalBaselineOffPAC
from src.agents.offpolicy.coagent_state_global_baseline import CoagentStateGlobalBaselineOffPAC

# greedy target policy
from src.agents.coagent_greedy import CoagentGreedy
from src.agents.coagent_greedy_global_baseline import CoagentGreedyGlobalBaseline

# discrete agents
from src.agents.coagent_discrete import CoagentDiscrete
from src.agents.coagent_discrete_global_baseline import CoagentDiscreteGlobalBaseline
from src.agents.coagent_discrete_ac import CoagentDiscreteAC
from src.agents.coagent_discrete_ac_offpolicy import CoagentDiscreteACOffPolicy

def get_agent(agent_name):
    if agent_name == 'backprop':
        return BackProp
    if agent_name == 'coagent':
        return Coagent
    if agent_name == 'coagent_global_baseline':
        return CoagentGlobalBaseline
    if agent_name == 'coagent_backprop':
        return CoagentBackprop
    if agent_name == 'coagent_global_baseline_backprop':
        return CoagentGlobalBaselineBackprop
    if agent_name == 'Monte' :
        return Monte_Regression
    if agent_name == 'Sarsa' :
        return Sarsa
    if agent_name == 'Sarsa_with_global_loss' :
        return Sarsa_with_global_loss
    if agent_name == 'Q' :
        return Q
    if agent_name == 'Q_with_global_loss' :
        return Q_with_global_loss
    if agent_name == 'Linear' :
        return Linear
    if agent_name == 'coagent_discrete_global_baseline':
        return CoagentDiscreteGlobalBaseline
    if agent_name == 'coagent_greedy':
        return CoagentGreedy
    if agent_name == 'coagent_greedy_global_baseline':
        return CoagentGreedyGlobalBaseline
    if agent_name == 'coagent_discrete':
        return CoagentDiscrete
    if agent_name == 'coagent_discrete_ac':
        return CoagentDiscreteAC 
    if agent_name == 'coagent_discrete_ac_offpolicy':
        return CoagentDiscreteACOffPolicy
    else:
        raise NotImplementedError

