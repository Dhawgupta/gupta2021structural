agent_colors = {
    # standard expeirments
    'backprop' : 'blue',
    'coagent' : 'orange',
    'coagent_global_baseline' : 'green',
    'coagent_state_global_baseline' : 'red',
    'coagent_state_layer_baseline' : 'purple',
    'coagent_state_layer_baseline_td' : 'cyan',
    'coagent_stochastic_backprop' : 'green',
    'coagent_coordinate_descent' : 'purple',
    'coagent_multiaction' : 'orange',
    'coagent_multiaction_global_baseline' : 'green',
    'Monte_Classifier' : 'red',

    # critic
    'coagent_ac' : 'purple',
    'coagent_ac_offpolicy' : 'black',
    'coagent_ac_layer_baseline' : 'magenta',

    'coagent_ac_offpolicy' : 'green',

    # entropy
    'coagent_ac_entropy' : 'green',

    'coagent_ac_offpolicy_entropy' : 'blue',

    # subset
    'coagent_subset' : 'green',
    'coagent_subset_global_baseline' : 'orange',

    # critic subset
    'coagent_ac_subset' : 'green',


    # debug experiments
    'coagent_backprop' : 'red',
    'coagent_global_baseline_backprop' : 'green',


    # off policy experiemtns
    'coagent_backprop_offpolicy' : 'cyan',
    'coagent_backprop_offpolicy2' : 'red',
    'coagent_global_baseline_backprop_offpolicy' : 'magenta',
    'coagent_global_baseline_backprop_offpolicy2' : 'purple',

    # value baseline global baseline
    'coagent_global_baseline_backprop_vf' : 'red',


    # off policy colors
    'coagent_offpac' : 'magenta',
    'coagent_global_baseline_offpac' : 'yellow',
    'coagent_state_global_baseline_offpac' : 'cyan',

    'coagent_ac_offpolicy_reinforce' : 'green',
    'coagent_ac_reinforce' : 'purple',

}


line_stype = {
    1 : '-',
    2 : '--',
    3 : ':'
}

line_type_nodes = {
    1 : '-',
    4 : '--',
    16 : ':',
    32: '--',
    63: ':',
    64 : '-.'
}

colors_nodes ={
    1: 'blue',
    4: 'green',
    16: 'red',
    32: 'orange',
    63: 'violet',
    64: 'black'

}

line_node = {
    'continuous' : '-',
    'discrete' : ':'
}

critic_model = {
    'linear' : '-',
    'network' : ':'
}

line_stype_pretrain = {
    'pretrain' : '-',
    'pretrain_layer' : '--',
    'pretrain_node' : ':',
    'pretrain_none' : '-*-',
    'pretrain_node_deterministic_node' : '-.'
}

line_type_pretrain = {
    True : '-',
    False : ':'
}

colors_partition = {
    's' : 'orange',
    'd' : 'blue',
    'ddd': 'blue',
    'sss' : 'orange',
    'sdd' : 'red',
    'dsd' : 'green',
    'dds' : 'magenta',

    'dd' : 'purple',
    'ss' : 'orange',
    'sd' : 'red',
    'ds' : 'green',
}
colors_partition_pretrain2 = {
    's' : 'red',
    'sd': 'blue',
    'ss' : 'green',
    'ds' : 'brown'
}

line_type_modelpart = {
    'backprop' : '-',
    'coagent' : ':'
}


# Off Pac stuff
colors_offpac = {

    'coagent' : 'orange',
    'coagent_global_baseline' : 'green',
    'coagent_state_layer_baseline' : 'purple',

    # Off policy colors
    'coagent_offpac' : 'orange',
    'coagent_global_baseline_offpac' : 'green',
    'coagent_state_global_baseline_offpac' : 'purple'


}


'''
color
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white
'''


'''

linestyle	description
'-' or 'solid'	solid line
'--' or 'dashed'	dashed line
'-.' or 'dashdot'	dash-dotted line
':' or 'dotted'	dotted line
'None'	: draw nothing
' '	draw nothing
''	draw nothing

'''