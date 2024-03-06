'''
This file will be used to plot learning curves for the combination of 2 keys
example :
`python analysis/learning_curve_key_key.py model_partition model_std experiments/debug.json
Plots each json file on a separete graph. 
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np

from src.utils.json_handling import get_sorted_dict, get_sorted_dict_loaded
from analysis.utils import find_best_key, smoothen_runs
from src.utils.formatting import get_folder_name, create_folder
from analysis.colors import colors_partition, line_type_pretrain

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/learning_curve_key_key.py json_file")
    exit()


assert sys.argv[1].lower() in ['y' ,'n'] , "[ERROR], Choose between y/n"
show_legend = sys.argv[1].lower() == 'y'
metric = sys.argv[2].lower()
assert metric in ['auc', 'last'], "[ERROR] wrong choice"
opt = sys.argv[3].lower()
assert opt in ['train', 'valid'], "[ERROR] wrong choice"
json_files = sys.argv[4:] # all the json files
key1 = "pretrain"
json_handles = [get_sorted_dict(j) for j in json_files]
key1 = 'model_partition'
key2 = 'pretrain'
# json_files = sys.argv[1:]
#
# json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)


def  plot(ax , data, label = None , color = None, line_style = None):
    mean =  data['mean'].reshape(-1)
    mean = smoothen_runs(mean)
    stderr =  data['stderr'].reshape(-1)
    base, = ax.plot(mean, label = label, linewidth = 2, color = color, linestyle = line_style)
    (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color = base.get_color(),  alpha = 0.4  )

key_to_plot = 'test' # the key to plot the data

fig, axs = plt.subplots(1, figsize=(6, 4), dpi=300)
axs = [axs]
for js in json_handles:
    # fig, axs = plt.subplots(1, figsize=(6, 4), dpi=300)
    # axs = [axs]
    runs, params, keys, data = find_best_key(js, key = [key1,key2], data = opt)
    # print(js)
    # agent_name = js['agent'][0]
    agent = params[keys[0]]['agent']
    layers = params[keys[0]]['model_specification']['num_layers']
    for k in keys: # pretrain : true, fals
        for i, key in enumerate(['test']):
            label = None
            if  k[1]:
                label = f'{k[0]}'

            plot(axs[i], data = data[k][key], label = label, color = colors_partition[k[0]], line_style= line_type_pretrain[k[1]] )
        # print(key_to_plot, data[key_to_plot]['mean'][-5:], data[key_to_plot]['stderr'][-5:])
        # axs[i].set_yscale('log')
            axs[i].set_ylim([0, 100])
            axs[i].spines['top'].set_visible(False)
            if show_legend:
                axs[i].set_title(f'{key} loss')
                axs[i].legend()

            axs[i].spines['right'].set_visible(False)
            axs[i].tick_params(axis='both', which='major', labelsize=8)
            axs[i].tick_params(axis='both',  which='minor', labelsize=8)
            axs[i].set_rasterized(True)

fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
# get_experiment_name = input("Give the input for experiment name: ")
layers = js['model_specification'][0]['num_layers']
print(layers)
# plt.savefig(f'{foldername}/learning_curve_{key1}_{key2}_{agent}_{layers}layers-{opt}-{metric}.pdf', dpi=300)
plt.savefig(f'{foldername}/learning_curve_{key1}_{key2}_{agent}_{layers}layers-{opt}-{metric}.png', dpi=300)