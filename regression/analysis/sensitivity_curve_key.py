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
from analysis.utils import find_best_key, smoothen_runs
from src.utils.formatting import get_folder_name, create_folder
from analysis.colors import agent_colors

# read the arguments etc
if len(sys.argv) < 3:
    print("usage : python analysis/plot_learning_curve.py key json_file")
    exit()

key = sys.argv[1]
json_files = sys.argv[2:]

# convert all json files to dict
json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)


def plot_sensitivity(ax, xaxis, data, label= None , stderr = False, color = None):
    data_list = []
    xaxis = sorted(xaxis)
    for k in xaxis:
        data_list.append(np.mean( data[k]['mean']))
    # print(xaxis, data_list)
    if color is not None:
        base, =  ax.plot(xaxis, data_list, '-*', label = label, color = color)



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

fig, axs = plt.subplots(1)
for js in json_handles:
    # runs, params, keys = find_best_key(js, key = key)
    d_keys = key
    runs, params, keys, best_data = find_best_key(js, key= d_keys, data = 'valid')
    print(keys)
    keys = sorted(keys)
    flipped = invert_keys(best_data)
    # print(flipped['test'])
    agent_name = params[keys[0]]['agent']
    for i, k in enumerate(flipped.keys()):
        if k == 'test':
            plot_sensitivity(axs, xaxis=keys, data=flipped[k], label = f"{agent_name}", color = agent_colors[agent_name])

axs.legend()
axs.set_xscale('log', basex=2)


# axs.set_ylim([50, 100])
axs.spines['top'].set_visible(False)


axs.spines['right'].set_visible(False)
axs.tick_params(axis='both', which='major', labelsize=8)
axs.tick_params(axis='both', which='minor', labelsize=8)
axs.set_rasterized(True)
fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
get_experiment_name = input("Give the input for experiment name: ")
plt.savefig(f'{foldername}/sensitivity_curve_{get_experiment_name}.pdf', dpi = 300)
plt.savefig(f'{foldername}/sensitivity_curve_{get_experiment_name}.png', dpi = 300)


