'''
Plot sensitviity curves with respect to 2 keys
key1 will be the key which differs on a single plot line and key 2 will be having multiple line
Status : INcomplete
'''
import os, sys, time, copy
import numpy as np
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from collections import defaultdict
from src.utils.json_handling import get_sorted_dict
from analysis.utils import find_best_key, smoothen_runs, find_best_key_subkeys
from src.utils.formatting import get_folder_name, create_folder
from analysis.colors import agent_colors

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/plot_learning_curve.py json_file")
    exit()

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
BIGGEST_SIZE = 25

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGEST_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGEST_SIZE)    # fontsize of the tick labels
# plt.rc('xtick', titlesize=BIGGEST_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', titlesize=BIGGEST_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

key1 = "units_layer"
key2 = "num_actions"
json_files = sys.argv[1:]

json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

# num_actions_colors = {
#     2 : 'black',
#     3 : 'purple',
#     5 : 'blue',
#     6 : 'red',
#     9 : 'green',
#     17 : 'orange'
# }

num_actions_colors = {
    2 : 'red',

    5 : 'blue',

    9 : 'green',
    17 : 'orange'
}

def plot_sensitivity(ax, xaxis, data, label= None , stderr = False, color = None, key2 = None):
    data_list = []
    xaxis = sorted(xaxis)
    for k in xaxis:
        data_list.append(np.mean(data[k]['mean'][-20:]))
    # print(xaxis, data_list)
    if color is not None:
        base, =  ax.plot(xaxis, data_list, '-', label = label, color = color, linewidth = 3)
    else:
        base, = ax.plot(xaxis, data_list, '-', label=label, linewidth = 3)
    # if key2 is not None:
    #     # index  = None
    #     try:
    #         index = xaxis.index(key2[0])
    #         ax.scatter(key2[0], data_list[index], marker = '*', color = base.get_color() )
    #     except:
    #         pass

def plot_entropy_layerr(ax, xaxis, data, label= None , stderr = False, color = None, key2 = None):
    layers = [1,2]
    layer_type = {
        1 : ':',
        2 : '--'
    }
    for l in layers:
        data_list = []
        xaxis = sorted(xaxis)
        for k in xaxis:
            data_list.append(np.mean(data[k]['mean'][-20:, l,:]))
        # print(xaxis, data_list)
        if color is not None:
            base, =  ax.plot(xaxis, data_list, layer_type[l], color = color, linewidth = 3)
        else:
            base, = ax.plot(xaxis, data_list,  layer_type[l], linewidth = 3)
        # if key2 is not None:
        #     # index  = None
        #     try:
        #         index = xaxis.index(key2[0])
        #         ax.scatter(key2[0], data_list[index], marker = '*', color = base.get_color() )
        #     except:
        #         pass


def get_parameter_data(data_all, keys, prefix_keys):
    data = dict()

    for k in keys:
        val = prefix_keys + [k]
        val = tuple(val)
        data[k] = data_all[val]
    return  data

def invert_keys(d):

    flipped = defaultdict(dict)
    for key, val in d.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    return flipped

# plot a single plot per json file
for js in json_handles:
    fig, axs = plt.subplots(1, figsize = (6,4))
    # runs, params, keys = find_best_key(js, key = key)
    # d_keys = key
    runs, params, keys, sub_keys,  best_data = find_best_key_subkeys(js, key= key1, subkeys =[ key2], data = 'valid')

    agent_name = js['agent'][0]
    keys = sorted(keys)
    print(keys)
    flipped = dict()
    for k in best_data.keys():
        flipped[k] = invert_keys(best_data[k])
    # print(flipped['test'])
    for i, k2 in enumerate(flipped.keys()):
        for j, data_type in enumerate(flipped[k2].keys()):
            if data_type == 'train':
                # label = None
                print("plotting")
                label = f'{key2}-{k2}'
                print(label)
                plot_sensitivity(axs, xaxis=keys, data=flipped[k2][data_type], label=label, key2 = k2, color = num_actions_colors[k2[0]])
            if data_type == 'entropy':
                # label = None
                print("plotting entrop")
                label = f'{key2}-{k2}'
                print(label)
                plot_entropy_layerr(axs, xaxis=keys, data=flipped[k2][data_type], label=label, key2 = k2, color = num_actions_colors[k2[0]])

    axs.legend()
    axs.set_xscale('log', basex=2)


    axs.set_ylim([0, 1])
    axs.spines['top'].set_visible(False)


    axs.spines['right'].set_visible(False)
    axs.tick_params(axis='both', which='major', labelsize=8)
    axs.tick_params(axis='both', which='minor', labelsize=8)
    axs.set_rasterized(True)
    fig.tight_layout()

    foldername = './plots'
    create_folder(foldername)
    # plt.legend()
    # plt.show()

    # get_experiment_name = input("Give the input for experiment name: ")
    get_experiment_name = agent_name
    plt.savefig(f'{foldername}/sensitivity_curve_{get_experiment_name}_{key1}_{key2}.png', dpi = 300)
    plt.savefig(f'{foldername}/sensitivity_curve_{get_experiment_name}_{key1}_{key2}.pdf', dpi=300)
    # plt.savefig(f'{foldername}/sensitivity_curve_{get_experiment_name}.pdf', dpi = 300)


