'''
This code will produce the learning curve for different agents
that are specified in the json files
Status : Complete (not completed the key based best parameter selection part)
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt

from src.utils.json_handling import get_sorted_dict
from analysis.utils import find_best, smoothen_runs, find_best_key
from src.utils.formatting import create_folder
from analysis.colors import agent_colors, line_type_pretrain

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
key1 = "pretrain"
json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

def  plot(ax , data, label = None , color = None, line_style = None):
    mean =  data['mean'].reshape(-1)
    mean = smoothen_runs(mean)
    # mean = smoothen_runs(mean)
    stderr =  data['stderr'].reshape(-1)
    base, = ax.plot(mean, label = label, linewidth = 2, color = color, linestyle = line_style)
    (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color = base.get_color(),  alpha = 0.1  )

key_to_plot = 'train' # the key to plot the data

fig, axs = plt.subplots(3, figsize = (6, 10 ), dpi = 300)
for en, js in enumerate(json_handles):
    runs, param, keys, data = find_best_key(js, key = [key1], data = opt, metric = metric)
    print(param)
    agent = param[keys[0]]['agent']
    layers = param[keys[0]]['model_specification']['num_layers']
    for k in keys: # pretrain : true false
        for i, key in enumerate(['train', 'valid','test']):
            label = None
            if not k[0]:
                label = f'{agent}'
            plot(axs[i], data = data[k][key], label = label, color = agent_colors[agent], line_style= line_type_pretrain[k[0]] )
        # print(key_to_plot, data[key_to_plot]['mean'][-5:], data[key_to_plot]['stderr'][-5:])
        # axs[i].set_yscale('log')
            axs[i].set_ylim([-5, 60])
            axs[i].spines['top'].set_visible(False)
            if show_legend:
                axs[i].set_title(f'{key} loss')
                # axs[i].legend()

            axs[i].spines['right'].set_visible(False)
            axs[i].tick_params(axis='both', which='major', labelsize=8)
            axs[i].tick_params(axis='both',  which='minor', labelsize=8)
            axs[i].set_rasterized(True)

#     for k in keys:
#         for i, dk in enumerate(data[k].keys()):
#             if dk in ['test']:
#                 label = None
#                 if k[0]: ## true
#                     label = f'{agent}'
#                 plot(axs, data=data[k][dk], label=label, color=agent_colors[agent],
#                      line_style= line_type_pretrain[k[0]])
#                 axs.set_title(f"{key1}")
# axs.set_ylim([0, 100])
# axs.spines['top'].set_visible(False)
# if show_legend:
#     axs.set_title(f'{key_to_plot} accuracy')
#     axs.legend()
#
# axs.spines['right'].set_visible(False)
# axs.tick_params(axis='both', which='major', labelsize=8)
# axs.tick_params(axis='both', which='minor', labelsize=8)
# axs.set_rasterized(True)
fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
# get_experiment_name = input("Give the input for experiment name: ")
# plt.savefig(f'{foldername}/learning_curve_compare-{layers}Layers-{opt}-{metric}.pdf', dpi = 300)
plt.savefig(f'{foldername}/learning_curve_compare-{layers}Layers-{opt}-{metric}.png', dpi = 300)


