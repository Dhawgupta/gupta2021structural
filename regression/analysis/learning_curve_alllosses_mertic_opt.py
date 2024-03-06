'''
This code will produce the learning curve for different agents
that are specified in the json files
Status : Complete (not completed the key based best parameter selection part)
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt

from src.utils.json_handling import get_sorted_dict
from analysis.utils import find_best, smoothen_runs
from src.utils.formatting import create_folder
from analysis.colors import agent_colors

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/learning_curve.py legend(y/n) <list of json files>")
    exit()

assert sys.argv[1].lower() in ['y' ,'n'] , "[ERROR], Choose between y/n"
show_legend = sys.argv[1].lower() == 'y'
metric = sys.argv[2].lower()
assert metric in ['auc', 'last'], "[ERROR] wrong choice"
opt = sys.argv[3].lower()
assert opt in ['train', 'valid'], "[ERROR] wrong choice"
json_files = sys.argv[4:] # all the json files

json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

def  plot(ax , data, label = None , color = None):
    mean =  data['mean'].reshape(-1)
    # mean = smoothen_runs(mean)
    mean = smoothen_runs(mean)
    stderr =  data['stderr'].reshape(-1)
    if color is not None:
        base, = ax.plot(mean, label = label, linewidth = 2, color = color)
    else:
        base, = ax.plot(mean, label=label, linewidth=2)
    (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color = base.get_color(),  alpha = 0.4  )

# key_to_plot = 'test' # the key to plot the data

fig, axs = plt.subplots(3, figsize = (6, 10 ), dpi = 300)
for en, js in enumerate(json_handles):
    run, param , data = find_best(js, data = opt , metric = metric)
    print(param)
    agent = param['agent']
    layers = param['model_specification']['num_layers']
    for i, key in enumerate(['train', 'valid','test']):
        plot(axs[i], data = data[key], label = f"{agent}", color = agent_colors[agent] )
    # print(key_to_plot, data[key_to_plot]['mean'][-5:], data[key_to_plot]['stderr'][-5:])
    # axs[i].set_yscale('log')
        axs[i].set_ylim([0, 50])
        axs[i].spines['top'].set_visible(False)
        if show_legend:
            axs[i].set_title(f'{key} accuracy')
            axs[i].legend()

        axs[i].spines['right'].set_visible(False)
        axs[i].tick_params(axis='both', which='major', labelsize=8)
        axs[i].tick_params(axis='both', which='minor', labelsize=8)
        axs[i].set_rasterized(True)


fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
# get_experiment_name = input("Give the input for experiment name: ")
get_experiment_name = 'noPretrain'
# plt.savefig(f'{foldername}/learning_curve_{get_experiment_name}-{layers}Layers-{opt}-{metric}.pdf', dpi = 300)
plt.savefig(f'{foldername}/learning_curve_{get_experiment_name}-{layers}Layers-{opt}-{metric}.png', dpi = 300)


