import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt


def plot_results(elapsed_training_steps, test_losses, train_losses, difficulty):
    plt.figure(figsize=(6,4), dpi=120)
    plt.plot(elapsed_training_steps, train_losses, label='train loss')
    plt.plot(elapsed_training_steps, test_losses, label='test loss')
    plt.xlabel('Elapsed training steps')
    plt.ylabel('MSE loss')
    plt.ylim((0, 1))
    plt.title(f'Difficulty $d={difficulty:0.2f}$')
    plt.legend()
    plt.show()



'''
This code will produce the learning curve for different agents
that are specified in the json files
Status : Complete (not completed the key based best parameter selection part)
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt

from src.utils.json_handling import get_sorted_dict
from analysis.correlated_utils import find_best, smoothen_runs
from src.utils.formatting import create_folder
from analysis.colors import agent_colors

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/learning_curve.py legend(y/n) <list of json files>")
    exit()

assert sys.argv[1].lower() in ['y' ,'n'] , "[ERROR], Choose between y/n"
show_legend = sys.argv[1].lower() == 'y'
json_files = sys.argv[2:] # all the json files

json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

def  plot(ax , data, label = None , color = None):
    mean =  data['mean'].reshape(-1)
    
    mean = smoothen_runs(mean)
    stderr = data['stderr'].reshape(-1)
    # 100 below is the train-test-ratio in correlated.py
    scaled_x = [i for i in range(1, mean.shape[0]*100, 100)]

    if color is not None:
        base, = ax.plot(scaled_x,mean, label = label, linewidth = 2, color = color)
    else:
        base, = ax.plot(scaled_x,mean, label=label, linewidth=2)
    (low_ci, high_ci) = confidence_interval(mean, stderr)

    ax.fill_between(scaled_x, low_ci, high_ci, color = base.get_color(),  alpha = 0.4  )

key_to_plot = 'test' # the key to plot the data

fig, axs = plt.subplots(1, figsize = (6, 4 ), dpi = 300)
for en, js in enumerate(json_handles):
    run, param , data = find_best(js, data = 'valid', metric = 'auc')
    print(param)
    agent = param['agent']
    plot(axs, data = data[key_to_plot], label = f"{agent}", color = agent_colors[agent] )
    # print(key_to_plot, data[key_to_plot]['mean'][-5:], data[key_to_plot]['stderr'][-5:])

# axs.set_ylim([0, 100])
axs.spines['top'].set_visible(False)
if show_legend:
    axs.set_title(f'{key_to_plot} Loss')
    axs.legend()

axs.spines['right'].set_visible(False)
axs.tick_params(axis='both', which='major', labelsize=8)
axs.tick_params(axis='both', which='minor', labelsize=8)
axs.set_rasterized(True)
fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
get_experiment_name = input("Give the input for experiment name: ")
plt.savefig(f'{foldername}/learning_curve_{get_experiment_name}.pdf', dpi = 300)
plt.savefig(f'{foldername}/learning_curve_{get_experiment_name}.png', dpi = 300)


