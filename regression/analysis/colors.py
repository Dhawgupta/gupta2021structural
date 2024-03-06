agent_colors = {
    # standard expeirments
    'backprop' : 'blue',
    'coagent' : 'orange',
    'coagent_global_baseline' : 'green',
    'coagent_state_global_baseline' : 'red',
    'coagent_state_layer_baseline' : 'purple',
    'coagent_state_layer_baseline_td' : 'cyan',


    # debug experiments
    'coagent_backprop' : 'orange',
    'coagent_global_baseline_backprop' : 'green',


    # off policy experiemtns
    'coagent_backprop_offpolicy' : 'cyan',
    'coagent_backprop_offpolicy2' : 'red',
    'coagent_global_baseline_backprop_offpolicy' : 'magenta',
    'coagent_global_baseline_backprop_offpolicy2' : 'purple',

    # value baseline global baseline
    'coagent_global_baseline_backprop_vf' : 'red',

    # action value

    'Monte' : 'green',
    'Q_with_global_loss' : 'red',
    'Sarsa_with_global_loss' : 'purple',

    # for correlated experiments
    'coagent_discrete_ac_offpolicy' : 'magenta',

}

line_stype = {
    1 : '-',
    2 : '--',
    3 : ':'
}

line_type_pretrain = {
    True : '-',
    False : ':'
}

colors_partition = {
    'ddd': 'blue',
    'sss' : 'orange',
    'sdd' : 'red',
    'dsd' : 'green',
    'dds' : 'magenta',
    'dd' : 'blue',
    'ss' : 'orange',
    'sd' : 'red',
    'ds' : 'green',
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