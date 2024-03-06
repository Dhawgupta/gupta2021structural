import torch


def compare_backprop_coagents(bp_agent, c_agent):
    '''
    Compare wether the specificatino of the bp agent and c_agent are same or not
    '''
    assert bp_agent.network.num_layers == c_agent.network.num_layers, "Number layers mismatch"
    assert bp_agent.network.num_nodes_layer == c_agent.network.num_coagents , "Layer width mismatch"
    assert bp_agent.network.activation == c_agent.network.activation, "Activation mismatch"
    # return True if same
    return True


# copy over the parameters from backprop model to cogaent mode
def copy_parameters(bp, co):
    with torch.no_grad():
        # compare_backprop_coagents(bp, co)
        for bp_param , co_param in zip(bp.network.parameters(), co.network.parameters()):
            co_param.data = bp_param.data
        # for i in range(bp.network.num_layers + 1):
        #     # copy the parameters
        #     # copy weights
        #     co.network.layers[i].weight.copy_(  bp.network.model[i * 2].weight)
        #     # copy biases
        #     co.network.layers[i].bias.copy_(bp.network.model[i * 2].bias)




