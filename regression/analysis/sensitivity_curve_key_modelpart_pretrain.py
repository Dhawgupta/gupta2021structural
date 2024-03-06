'''
These will plot senstivirty file for alll teh algorihtms for the alpha with repsect to a given key
Status : INcomplete
'''
import os, sys, time, copy
import numpy as np
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from collections import defaultdict
from src.utils.json_handling import get_sorted_dict
from analysis.utils import find_best_key_subkeys, smoothen_runs
from src.utils.formatting import get_folder_name, create_folder
from analysis.colors import colors_partition, line_type_pretrain

# read the arguments etc
if len(sys.argv) < 3:
    print("usage : python analysis/sensitvity_curve_key_modelpart_pretrain.py key json_file")
    exit()

key = sys.argv[1]
key1 = 'model_partition'
key2 = 'pretrain'
json_files = sys.argv[2:]

# convert all json files to dict
json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)


def plot_sensitivity(ax, xaxis, data, label= None , stderr = False, color = None, line_style = None):
    data_list = []
    xaxis = sorted(xaxis)
    for k in xaxis:
        data_list.append(np.mean(100* data[k]['mean']))
    # print(xaxis, data_list)
    if color is not None:
        base, =  ax.plot(xaxis, data_list, '-*', label = label, color = color, linestyle = line_style)



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


for js in json_handles:
    fig, axs = plt.subplots(1)
    d_keys = key
    runs, params, keys, subkeys, best_data = find_best_key_subkeys(js, key= key, subkeys = [key1 , key2],  data = 'valid')

    keys = sorted(keys)
    flipped = dict()
    for k in best_data.keys():
        flipped[k] = invert_keys(best_data[k])

    for i, sk in enumerate(flipped.keys()):
        for j, k in enumerate(flipped[sk].keys()):
            if k == 'test':
                label = None
                if sk[1]:
                    label = f'{sk[0]}'
                plot_sensitivity(axs, xaxis=keys, data=flipped[sk][k], label = label, color = colors_partition[sk[0]],  line_style=line_type_pretrain[sk[1]])

    axs.legend()
    axs.set_xscale('log', basex=2)


    axs.set_ylim([50, 100])
    axs.spines['top'].set_visible(False)


    axs.spines['right'].set_visible(False)
    axs.tick_params(axis='both', which='major', labelsize=8)
    axs.tick_params(axis='both', which='minor', labelsize=8)
    axs.set_rasterized(True)
    fig.tight_layout()

    foldername = './plots'
    create_folder(foldername)
    # plt.legend()
    # get_experiment_name = input("Give the input for experiment name: ")
    agent_name = js['agent'][0]
    layers = js['model_specification'][0]['num_layers']
    # plt.savefig(f'{foldername}/sensitivity_curve_{key1}_{key2}_{agent_name}_{layers}layers.pdf', dpi = 300)
    plt.savefig(f'{foldername}/sensitivity_curve_{key}_{key1}_{key2}_{agent_name}_{layers}layers.png', dpi = 300)


