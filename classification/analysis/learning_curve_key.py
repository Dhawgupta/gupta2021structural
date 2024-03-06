'''
These work for single json file where the comparison is done accross a given key in the json file
Status : Working
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np

from src.utils.json_handling import get_sorted_dict, get_sorted_dict_loaded
from analysis.utils import find_best_key, smoothen_runs
from src.utils.formatting import get_folder_name, create_folder
from analysis.utils import load_different_runs, pkl_saver, pkl_loader
import json

# read the arguments etc
if len(sys.argv) < 3:
    print("usage : python analysis/plot_learning_curve.py key json_file")
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

key = sys.argv[1]
json_files = sys.argv[2:]

json_handles = [get_sorted_dict(j) for j in json_files]

import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
BIGGEST_SIZE = 25

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

smoothfactor = 0.9
def  plot(ax , data, label = None ):
    mean = 100 * data['mean'].reshape(-1)
    mean = smoothen_runs(mean, smoothfactor)
    stderr = 100 * data['stderr'].reshape(-1)
    base, = ax.plot(mean, label = label, linewidth = 3)
    (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color = base.get_color(),  alpha = 0.4  )

key_to_plot = 'test' # the key to plot the data


for js in json_handles:
    fig, axs = plt.subplots(1, figsize=(6, 4), dpi=300)
    runs, params, keys, data = find_best_key(js, key = key, data = 'valid')
    for k in keys:
        for i, dk in enumerate(data[k].keys()):
            if dk in ['test']:
                print(params[k])
                plot(axs, data = data[k][dk], label = f'{k}')
                axs.set_title(f"{key}")
    axs.legend()
    axs.set_ylim([0, 100])
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
    plt.savefig(f'{foldername}/learning_curve_{key}_{get_experiment_name}.pdf', dpi=300)
    plt.savefig(f'{foldername}/learning_curve_{key}_{get_experiment_name}.png', dpi=300)