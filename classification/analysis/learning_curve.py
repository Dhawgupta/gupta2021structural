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
from analysis.colors import agent_colors, line_node, critic_model


# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/learning_curve.py legend(y/n) <list of json files>")
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

# assert sys.argv[1].lower() in ['y' ,'n'] , "[ERROR], Choose between y/n"

show_legend = True
metric = "auc"
json_files = sys.argv[1:] # all the json files

json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

def  plot(ax , data, label = None , color = None,  line_style = None):
    if line_style is None:
        line_style = '-'
    mean = 100 * data['mean'].reshape(-1)
    # mean = smoothen_runs(mean)
    stderr = 100 * data['stderr'].reshape(-1)
    if color is not None:
        base, = ax.plot(mean, label = label, linewidth = 3, color = color, linestyle = line_style)
    else:
        base, = ax.plot(mean, label=label, linewidth=3, linestyle = line_style)
    (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color = base.get_color(),  alpha = 0.2  )

key_to_plot = 'train' # the key to plot the data

fig, axs = plt.subplots(1, figsize = (6, 4 ), dpi = 300)
for en, js in enumerate(json_handles):
    run, param , data = find_best(js, data = 'valid', metric = metric)
    print(param)
    agent = param['agent']
    critic_layer = int(param.get('critic_layer',0))
    if critic_layer == 0:
        critic_model_type = 'linear'
    elif critic_layer > 0:
        critic_model_type = 'network'
    # plot(axs, data = data[key_to_plot], label = f"{agent}", color = agent_colors[agent], line_style = critic_model[critic_model_type] )
    plot(axs, data = data[key_to_plot], label = f"{agent}", color = agent_colors[agent])
    # print(key_to_plot, data[key_to_plot]['mean'][-5:], data[key_to_plot]['stderr'][-5:])

axs.set_ylim([50, 90])
axs.spines['top'].set_visible(False)
if show_legend:
    axs.set_title(f'{key_to_plot} accuracy')
    axs.legend()

axs.spines['right'].set_visible(False)
axs.set_xlabel('Epochs') 
axs.tick_params(axis='both', which='major', labelsize=12)
axs.tick_params(axis='both', which='minor', labelsize=12)
axs.set_rasterized(True)
fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
# get_experiment_name = input("Give the input for experiment name: ")
get_experiment_name = 'l'
plt.savefig(f'{foldername}/learning_curve_{get_experiment_name}.pdf', dpi = 300)
plt.savefig(f'{foldername}/learning_curve_{get_experiment_name}.png', dpi = 300)


